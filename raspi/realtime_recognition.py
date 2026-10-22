#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webカメラの動画からリアルタイムに手書き数字を認識する（EV3なしで試す用）

使い方:
  uv run python realtime_recognition.py            # 画面があればウィンドウ、なければブラウザ配信
  uv run python realtime_recognition.py --web      # ブラウザ配信 (http://<PiのIP>:8000/)
  uv run python realtime_recognition.py --no-gui   # 文字だけで表示

キー（ウィンドウ表示のとき）:
  q : 終了
  s : 現在のフレームを photos/ に保存
"""
import argparse
import os
import subprocess
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from camera import load_model, preprocess_image, predict_digit


class LatestFrameReader:
    """別スレッドでカメラを読み続け、常に最新の1フレームだけを保持する

    推論が撮影より遅くても、古いフレームがたまって映像が遅れることがない。
    """

    def __init__(self, camera_id, width, height):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = self.cap.isOpened()
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1

    def read(self):
        with self.lock:
            return self.seq, self.frame

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()


PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Digit Recognition</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:#111;display:flex;justify-content:center">
<img src="/stream" style="width:100%;max-width:960px">
</body></html>"""


class MjpegServer:
    """認識結果を描いた映像を、ブラウザで見られる MJPEG として配信する"""

    def __init__(self, port):
        self.jpeg = None
        self.seq = 0
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    sent_seq = 0
                    try:
                        while True:
                            with server.lock:
                                seq, jpeg = server.seq, server.jpeg
                            if jpeg is None or seq == sent_seq:
                                time.sleep(0.01)
                                continue
                            sent_seq = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: %d\r\n\r\n" % len(jpeg))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def update(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.seq += 1

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def center_roi(frame, roi_size):
    h, w = frame.shape[:2]
    s = min(roi_size, w, h)
    x1 = (w - s) // 2
    y1 = (h - s) // 2
    return x1, y1, x1 + s, y1 + s


def draw_overlay(frame, roi, resized, stable, pred, conf, fps):
    """枠・認識結果・AIが見ている画像をフレームに描き込む"""
    x1, y1, x2, y2 = roi
    color = (0, 255, 0) if stable is not None else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{stable}" if stable is not None else "-"
    cv2.putText(frame, f"Result: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    raw = f"pred={pred} {conf:.1f}%" if pred is not None else "no digit"
    cv2.putText(frame, f"{raw}  fps={fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # AIが見ている28x28の画像を右上に拡大して表示
    view = cv2.resize(resized, (112, 112), interpolation=cv2.INTER_NEAREST)
    w = frame.shape[1]
    frame[0:112, w - 112:w] = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)


def local_ips():
    try:
        return subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="リアルタイム手書き数字認識")
    parser.add_argument("--camera", type=int, default=None,
                        help="カメラ番号 (/dev/videoN)。省略すると自動で探す")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--roi", type=int, default=300, help="認識エリアの一辺(px)")
    parser.add_argument("--conf", type=float, default=90.0, help="確信度の閾値(%%)")
    parser.add_argument("--smooth", type=int, default=5, help="多数決に使う直近の推論回数")
    parser.add_argument("--web", action="store_true", help="ブラウザに映像を配信する")
    parser.add_argument("--port", type=int, default=8000, help="ブラウザ配信のポート番号")
    parser.add_argument("--no-gui", action="store_true", help="映像を出さずに文字だけで表示")
    args = parser.parse_args()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gui = not args.no_gui and not args.web and has_display
    # 画面がない（SSH接続など）ときは、自動でブラウザ配信にする
    web = not args.no_gui and (args.web or not has_display)

    model = load_model("mnist_model.keras")
    if model is None:
        return

    # 最初の1回は準備に時間がかかるので、先に済ませておく
    predict_digit(model, np.zeros((1, 28, 28, 1), np.float32))

    # USBを挿し直すと番号が変わることがあるので、指定がなければ0から順に試す
    candidates = [args.camera] if args.camera is not None else range(10)
    for camera_id in candidates:
        if args.camera is None and not os.path.exists(f"/dev/video{camera_id}"):
            continue
        reader = LatestFrameReader(camera_id, args.width, args.height)
        if reader.running:
            print(f"カメラ {camera_id} (/dev/video{camera_id}) を使います。")
            break
        reader.release()
    else:
        print("エラー: カメラを開けませんでした。"
              " USB接続と ls /dev/video* を確認してください。")
        return

    streamer = None
    if web:
        try:
            streamer = MjpegServer(args.port)
        except OSError as e:
            print(f"エラー: ポート {args.port} で配信を始められませんでした ({e})。"
                  " --port で別の番号を指定してください。")
            reader.release()
            return
        print("ブラウザで次のどれかを開くと映像が見られます:")
        for ip in local_ips():
            if ":" not in ip:
                print(f"  http://{ip}:{args.port}/")

    print("認識を開始しました。" + ("'q' キーで終了します。" if gui else "Ctrl+C で終了します。"))

    history = deque(maxlen=args.smooth)
    stable = None
    last_seq = 0
    fps = 0.0
    last_t = time.time()
    last_log = 0.0

    try:
        while True:
            seq, frame = reader.read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq

            roi = center_roi(frame, args.roi)
            x1, y1, x2, y2 = roi
            input_data, resized = preprocess_image(frame[y1:y2, x1:x2])

            if resized.any():
                pred, conf = predict_digit(model, input_data)
            else:
                pred, conf = None, 0.0

            # 直近の結果で多数決し、その数字の平均確信度で判定してちらつきを抑える
            history.append((pred, conf))
            value, count = Counter(p for p, _ in history).most_common(1)[0]
            avg_conf = sum(c for p, c in history if p == value) / count
            new_stable = value if (value is not None and count > len(history) // 2
                                   and avg_conf >= args.conf) else None

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            if new_stable != stable:
                stable = new_stable
                print(f"認識結果: {stable if stable is not None else '-'}"
                      f"  (pred={pred} conf={conf:.1f}%)", flush=True)
            elif not gui and now - last_log >= 2.0:
                print(f"  fps={fps:.1f} pred={pred} conf={conf:.1f}%", flush=True)
                last_log = now

            if gui or streamer:
                draw_overlay(frame, roi, resized, stable, pred, conf, fps)

            if streamer:
                streamer.update(frame)

            if gui:
                cv2.imshow("Realtime Digit Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    os.makedirs("photos", exist_ok=True)
                    filename = os.path.join("photos", f"img_{int(time.time())}.jpg")
                    cv2.imwrite(filename, frame)
                    print("保存しました:", filename)

    except KeyboardInterrupt:
        pass
    finally:
        if streamer:
            streamer.close()
        reader.release()
        if gui:
            cv2.destroyAllWindows()
        print("終了しました。")


if __name__ == "__main__":
    main()
