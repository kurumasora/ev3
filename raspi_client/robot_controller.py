#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi側のTCPクライアント．カメラ → 推論 → EV3へ速度指令を送る．

通信形式（CLAUDE.md の「通信形式」と同じ）:
  EV3 → Pi : "distance_mm,gyro_angle\\n"
  Pi → EV3 : "left_speed,right_speed\\n"（整数，単位は度/秒）

EV3がセンサ値を1行送るたびに，Piが速度指令を1行返す．EV3は指令が1秒届かないと
モータを止めるので，推論が詰まっても暴走しない．

スレッドの構成:
  カメラスレッド（camera.py）: 撮影を続け，最新の1フレームだけを保持する．
  推論スレッド（DetectionWorker）: 最新のフレームを推論し続け，最新の結果だけを保持する．
  メインスレッド（run_session）: センサ値を受け取るたびに，最新の結果から速度を決めてすぐ返す．
  推論の終わりを待たないので，推論が遅くなっても通信は止まらない．

現在の動作:
  的の数字（TARGET_DIGITS）が見えている間だけ前進し，見えなくなったら止まる．
  1 → 2 → 3 の順番の管理や的の探索はまだ入っていない．decide_speeds() に追加する．

使い方（モータが動くので，ロボットを安全な状態にしてから実行する）:
  1. EV3で motor_sensor_server.py を起動する．
  2. cd raspi_client && uv run python robot_controller.py
  IPアドレスが変わったときは --host <EV3のIP> で指定する．
"""
import argparse
import socket
import threading
import time

import numpy as np

from camera import LatestFrameReader, center_roi
from digit_recognizer import load_model, predict_digit, preprocess_image

# ---- 通信 ----
EV3_HOST = "10.42.0.175"
EV3_PORT = 9999
CONNECT_TIMEOUT_S = 5.0   # 1回の接続の試行で待つ時間
RETRY_S = 2.0             # 接続に失敗したり切断されたりした後，再接続するまでの時間
RECV_TIMEOUT_S = 3.0      # EV3は1秒以内に必ず送ってくるので，これを超えたら切断とみなす

# ---- カメラ・推論 ----
MODEL_PATH = "mnist_model.keras"
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_SIZE = 400            # 画面中央の認識エリアの一辺(px)
CONF_TH = 95.0            # この確信度(%)以上のときだけ数字が見えたとみなす
TARGET_DIGITS = {1, 2, 3}
RESULT_TIMEOUT_S = 1.0    # 検出結果がこの時間更新されなければ，安全のため止まる（カメラや推論の停止）
WORKER_IDLE_S = 0.005     # 新しいフレームがないとき，推論スレッドが次に確認するまで待つ時間

# ---- 判断 ----
NEED_GO_FRAMES = 3        # この回数続けて見えたら前進を始める
NEED_STOP_FRAMES = 3      # この回数続けて見えなければ止まる
FORWARD_SPEED = 200       # 前進するときのモータ速度(度/秒)

# ---- ログ ----
LOG_EVERY_N = 30          # 指令をこの回数送るごとに状態を表示する


class GoStopDecider:
    """推論のちらつきで動作が切り替わらないように，連続回数で GO / STOP を決める"""

    def __init__(self):
        self.state = "STOP"
        self.seen = 0
        self.not_seen = 0

    def update(self, visible):
        if visible:
            self.seen += 1
            self.not_seen = 0
        else:
            self.not_seen += 1
            self.seen = 0

        if self.state == "STOP" and self.seen >= NEED_GO_FRAMES:
            self.state = "GO"
        elif self.state == "GO" and self.not_seen >= NEED_STOP_FRAMES:
            self.state = "STOP"

    def reset(self):
        self.state = "STOP"
        self.seen = 0
        self.not_seen = 0


def detect_digit(model, frame):
    """フレーム中央から数字を探し，(予測した数字, 確信度, 的が見えたか) を返す"""
    x1, y1, x2, y2 = center_roi(frame, ROI_SIZE)
    try:
        input_data, resized = preprocess_image(frame[y1:y2, x1:x2])
        # 真っ黒な画像 = 文字なし．推論すると何かの数字と誤判定するおそれがある
        if not resized.any():
            return None, 0.0, False
        pred, conf = predict_digit(model, input_data)
    except Exception as e:
        print("[RasPi] 推論エラー:", e, flush=True)
        return None, 0.0, False
    return pred, conf, conf >= CONF_TH and pred in TARGET_DIGITS


class DetectionWorker:
    """別スレッドで推論を続け，最新の検出結果だけを保持する

    通信ループが推論の終わりを待たずに済むので，推論が遅くても指令をすぐ返せる．
    結果には推論を終えた時刻を付け，カメラや推論が止まったときに古い結果で走り続けないようにする．
    推論に時間がかかるほど，結果が表す映像はその分だけ古い点に注意する．
    """

    def __init__(self, model, reader):
        self.model = model
        self.reader = reader
        self.lock = threading.Lock()
        # (結果の番号, 推論を終えた時刻, 予測した数字, 確信度, 的が見えたか)．まだ結果がなければ None
        self.result = None
        self.count = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        last_seq = 0
        while self.running:
            seq, frame = self.reader.read()
            # 前回と同じフレームなら推論しない
            if frame is None or seq == last_seq:
                time.sleep(WORKER_IDLE_S)
                continue
            last_seq = seq
            pred, conf, visible = detect_digit(self.model, frame)
            with self.lock:
                self.count += 1
                self.result = (self.count, time.monotonic(), pred, conf, visible)

    def read(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


def decide_speeds(state, distance_mm, gyro_angle):
    """状態とセンサ値から (左の速度, 右の速度) を決める

    distance_mm と gyro_angle はまだ使っていない．的への接近や旋回を
    追加するときにここで使う．
    """
    if state == "GO":
        return FORWARD_SPEED, FORWARD_SPEED
    return 0, 0


def parse_sensor_line(line):
    """ "distance_mm,gyro_angle" を (距離, 角度) にする．壊れた行なら None を返す"""
    try:
        distance_mm, gyro_angle = map(int, line.split(","))
    except ValueError:
        return None
    return distance_mm, gyro_angle


def receive_latest_line(conn, buffer):
    """改行まで届いた行のうち最新の1行と，残りのバッファを返す

    TCPでは行が途中で分割されて届くことがあるので，改行がない部分は
    次回までバッファに残し，途切れた値を使わないようにする．
    """
    while b"\n" not in buffer:
        data = conn.recv(1024)
        if not data:
            raise ConnectionError("EV3が接続を閉じました")
        buffer += data
    complete, _, rest = buffer.rpartition(b"\n")
    return complete.rsplit(b"\n", 1)[-1].decode("utf-8", "replace"), rest


def send_speeds(conn, left, right):
    conn.sendall("{},{}\n".format(left, right).encode("utf-8"))


def connect_ev3(host, port):
    """EV3に接続できるまで再試行する"""
    while True:
        print(f"[RasPi] EV3 ({host}:{port}) に接続します", flush=True)
        try:
            conn = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)
        except OSError as e:
            print(f"[RasPi] 接続できませんでした: {e}", flush=True)
            time.sleep(RETRY_S)
            continue
        conn.settimeout(RECV_TIMEOUT_S)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[RasPi] EV3に接続しました", flush=True)
        return conn


def run_session(conn, worker):
    """1回の接続の間，センサ値を受け取るたびに速度指令を返す．切断時は OSError が出る

    推論はしない．推論スレッドが置いた最新の結果を使うだけなので，すぐに指令を返せる．
    """
    decider = GoStopDecider()
    buffer = b""
    last_count = 0
    pred, conf = None, 0.0
    sent_count = 0

    while True:
        line, buffer = receive_latest_line(conn, buffer)
        previous_state = decider.state

        # 推論スレッドの最新の結果を使う．前回と同じ結果なら，回数を加算しない
        result = worker.read()
        now = time.monotonic()
        if result is None or now - result[1] >= RESULT_TIMEOUT_S:
            # カメラや推論が止まると古い判断で走り続けてしまうので止める
            decider.reset()
        elif result[0] != last_count:
            last_count, _, pred, conf, visible = result
            decider.update(visible)

        sensor = parse_sensor_line(line)
        if sensor is None:
            print(f"[RasPi] 壊れたセンサ値を無視します: {line!r}", flush=True)
            left, right = 0, 0
        else:
            left, right = decide_speeds(decider.state, *sensor)
        send_speeds(conn, left, right)
        sent_count += 1

        if decider.state != previous_state or sent_count % LOG_EVERY_N == 0:
            print(
                f"[RasPi] state={previous_state}->{decider.state} speed={left},{right} "
                f"sensor={sensor} pred={pred} conf={conf:.1f}%",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description="EV3へ接続して速度指令を送るクライアント")
    parser.add_argument("--host", default=EV3_HOST, help="EV3のIPアドレス")
    parser.add_argument("--port", type=int, default=EV3_PORT, help="EV3のポート番号")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    if model is None:
        print("[RasPi] モデルを読み込めませんでした", flush=True)
        return

    # 最初の推論は時間がかかり，EV3の1秒のタイムアウトを超えるおそれがあるので先に済ませる
    predict_digit(model, np.zeros((1, 28, 28, 1), np.float32))

    reader = LatestFrameReader(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
    if not reader.running:
        print(f"[RasPi] カメラ {CAMERA_ID} を開けませんでした", flush=True)
        reader.release()
        return

    # 接続する前から推論を始めておき，接続したらすぐ最新の結果を使えるようにする
    worker = DetectionWorker(model, reader)

    try:
        while True:
            conn = connect_ev3(args.host, args.port)
            try:
                run_session(conn, worker)
            except OSError as e:
                print(f"[RasPi] 通信が切れました: {e}", flush=True)
            finally:
                # 終了時や切断時に，届くならモータの停止を指令してから閉じる
                try:
                    send_speeds(conn, 0, 0)
                except OSError:
                    pass
                conn.close()
            time.sleep(RETRY_S)
    except KeyboardInterrupt:
        print("[RasPi] 終了します", flush=True)
    finally:
        worker.stop()
        reader.release()


if __name__ == "__main__":
    main()
