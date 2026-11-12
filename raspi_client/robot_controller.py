#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi側のTCPクライアント（本番で起動するファイル）．
カメラ → 推論 → 判断 → EV3へ指令を送る．

ファイルの分担:
  robot_controller.py : 起動，EV3との接続，通信のループ（このファイル）
  strategy.py         : 判断（探索 → 照準 → 接近 → 発射 → 後退 → 次の数字）
  protocol.py         : 通信形式の組み立てと解析
  digit_recognizer.py : 画面全体から数字を探して推論する（find_digits）
  camera.py           : カメラからのフレーム取得

EV3がセンサ値を1行送るたびに，Piが指令を1行返す．EV3は指令が1秒届かないと
走行モータを止めるので，Pi側が止まっても暴走しない．

スレッドの構成:
  カメラスレッド（camera.py）: 撮影を続け，最新の1フレームだけを保持する．
  推論スレッド（DetectionWorker）: 最新のフレームから数字を探し続け，最新の結果だけを保持する．
  メインスレッド（run_session）: センサ値を受け取るたびに，最新の結果から指令を決めてすぐ返す．
  推論の終わりを待たないので，推論が遅くなっても通信は止まらない．

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

from camera import LatestFrameReader
from digit_recognizer import find_digits, load_model, predict_digit
from protocol import format_command, parse_sensor_line, receive_latest_line
from strategy import STOP, Strategy

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
RESULT_TIMEOUT_S = 1.0    # 検出結果がこの時間更新されなければ，安全のため止まる（カメラや推論の停止）
WORKER_IDLE_S = 0.005     # 新しいフレームがないとき，推論スレッドが次に確認するまで待つ時間

# ---- ログ ----
LOG_EVERY_N = 30          # 指令をこの回数送るごとに状態を表示する


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
        # (結果の番号, 推論を終えた時刻, 検出結果のリスト)．まだ結果がなければ None
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
            try:
                detections = find_digits(self.model, frame)
            except Exception as e:
                print("[RasPi] 推論エラー:", e, flush=True)
                detections = []
            with self.lock:
                self.count += 1
                self.result = (self.count, time.monotonic(), detections)

    def read(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


class Commander:
    """指令を送る．発射番号（fire_id）を覚えておき，発射するときだけ1増やす

    EV3と再接続しても発射番号を引き継ぐ．EV3は接続直後に届いた番号を基準にするので，
    再接続しただけでは撃たない．
    """

    def __init__(self):
        self.fire_id = 0

    def send(self, conn, action):
        if action.fire:
            self.fire_id += 1
        conn.sendall(format_command(action.left, action.right, self.fire_id))


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


def summarize(detections):
    """ログ用に，検出結果を短い文字列にする"""
    return " ".join("{}({:.0f}%,{:+.2f})".format(d.digit, d.conf, d.x_offset) for d in detections[:3]) or "なし"


def run_session(conn, worker, strategy, commander):
    """1回の接続の間，センサ値を受け取るたびに指令を返す．切断時は OSError が出る

    推論はしない．推論スレッドが置いた最新の結果を使うだけなので，すぐに指令を返せる．
    """
    buffer = b""
    sent_count = 0

    while True:
        line, buffer = receive_latest_line(conn, buffer)
        now = time.monotonic()

        # カメラや推論が止まっていたら，結果を渡さない（strategy が止まる）
        result = worker.read()
        if result is not None and now - result[1] >= RESULT_TIMEOUT_S:
            result = None

        sensor = parse_sensor_line(line)
        if sensor is None:
            print(f"[RasPi] 壊れたセンサ値を無視します: {line!r}", flush=True)
            action = STOP
        else:
            action = strategy.step(now, sensor[0], sensor[1], result)
        commander.send(conn, action)
        sent_count += 1

        if sent_count % LOG_EVERY_N == 0:
            detections = summarize(result[2]) if result is not None else "結果なし（停止中）"
            print(
                f"[RasPi] 状態={strategy.state} 狙う数字={strategy.target} 指令={action.left},{action.right} "
                f"センサ={sensor} 検出={detections}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description="EV3へ接続して指令を送るクライアント")
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
    # 再接続しても，どの数字まで撃ったかと発射番号を引き継ぐ
    strategy = Strategy()
    commander = Commander()

    try:
        while True:
            conn = connect_ev3(args.host, args.port)
            try:
                run_session(conn, worker, strategy, commander)
            except OSError as e:
                print(f"[RasPi] 通信が切れました: {e}", flush=True)
            finally:
                # 終了時や切断時に，届くならモータの停止を指令してから閉じる
                try:
                    commander.send(conn, STOP)
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
