# 仮のフィールドで，本番のプログラム（robot_controller + strategy + find_digits）を通しで動かす．
# モータもカメラも使わない．EV3の代わりに，このファイルの中の仮のEV3サーバ（localhost）を使う．
#   - ロボットは中央からスタートし，周りに 1, 2, 3 の的がある（数字はスタート位置を向く）
#   - 仮のカメラは，ロボットの位置と向きから，的がどう見えるかを描く
#   - 仮のEV3は motor_sensor_server.py と同じ約束事で動く（指令でロボットを動かし，
#     発射番号が変わったら，正面の近くにある的に当たったかを記録する）
# 1 → 2 → 3 の順に当たり，最後に DONE になれば成功．
# 使い方：  cd raspi_client && uv run python test/sim_field.py
import math
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

# 親フォルダ（raspi_client/）のファイルを import し，モデルを相対パスで読み込めるようにする
CLIENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLIENT_DIR)
os.chdir(CLIENT_DIR)

import robot_controller as rc  # noqa: E402
import strategy as st  # noqa: E402

# ---- 仮のロボットとフィールド ----
PORT = 19998                              # 仮のEV3サーバのポート（本物と重ならない番号）
DURATION_S = 120                          # 最長の実行時間(秒)
WHEEL_MM_PER_DEG = math.pi * 56 / 360     # モータ1度あたりに進む距離（車輪の直径56mm）
TRACK_MM = 120                            # 左右の車輪の間隔(mm)
DIGIT_MM = 120                            # 印刷した数字の高さ(mm)
US_HALF_ANGLE_DEG = 12                    # 超音波センサが的をとらえる範囲（正面から左右に）
HIT_HALF_ANGLE_DEG = 4                    # 発射したときに当たったとみなす範囲（正面から左右に）
HIT_MAX_MM = 600                          # 当たったとみなす最大の距離
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FOCAL_PX = (FRAME_WIDTH / 2) / math.tan(math.radians(st.CAMERA_HFOV_DEG / 2))

# 的：数字と位置（x は右，y は前，単位 mm）．スタート時のロボットは y の正の向きを向く
TARGETS = [
    {"digit": "1", "x": -600, "y": 450, "up": True},
    {"digit": "2", "x": 650, "y": 380, "up": True},
    {"digit": "3", "x": 80, "y": -700, "up": True},
]

lock = threading.Lock()
robot = {"x": 0.0, "y": 0.0, "h": 0.0}     # h は向き(度)．時計回りが正．0 = y の正の向き
hits = []                                  # 発射の記録


def bearing_and_distance(target):
    """ロボットから見た的の向き（正面から何度右か）と距離(mm)"""
    dx, dy = target["x"] - robot["x"], target["y"] - robot["y"]
    bearing = math.degrees(math.atan2(dx, dy)) - robot["h"]
    bearing = (bearing + 180) % 360 - 180
    return bearing, math.hypot(dx, dy)


def render():
    """ロボットのカメラに映る画像を描く"""
    img = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 190, np.uint8)
    with lock:
        views = [(t, *bearing_and_distance(t)) for t in TARGETS if t["up"]]
    cy = FRAME_HEIGHT // 2
    for target, bearing, distance in views:
        if abs(bearing) > st.CAMERA_HFOV_DEG / 2 - 3:
            continue
        cx = int(FRAME_WIDTH / 2 + FOCAL_PX * math.tan(math.radians(bearing)))
        size = int(FOCAL_PX * DIGIT_MM / distance)
        pw, ph = int(size * 0.9), int(size * 1.4)
        cv2.rectangle(img, (cx - pw // 2, cy - ph // 2), (cx + pw // 2, cy + ph // 2), (250, 250, 250), -1)
        scale, thick = size / 22.0, max(2, int(size / 9))
        (tw, th), _ = cv2.getTextSize(target["digit"], cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, target["digit"], (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick)
    return img


class SimCamera:
    """camera.py の LatestFrameReader の代わり．1秒に30回描いて，最新の1枚だけを保持する"""

    def __init__(self):
        self.seq, self.frame, self.running = 0, None, True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            frame = render()
            with self.lock:
                self.seq += 1
                self.frame = frame
            time.sleep(1 / 30)

    def read(self):
        with self.lock:
            return self.seq, self.frame


def ultrasonic():
    """正面の近くにある的までの距離．何もなければ 2550"""
    best = 2550
    for target in TARGETS:
        if target["up"]:
            bearing, distance = bearing_and_distance(target)
            if abs(bearing) <= US_HALF_ANGLE_DEG:
                best = min(best, int(distance))
    return best


def record_shot(elapsed):
    """発射したときに，正面の近くにある的を倒す"""
    hit = None
    for target in TARGETS:
        bearing, distance = bearing_and_distance(target)
        if target["up"] and abs(bearing) <= HIT_HALF_ANGLE_DEG and distance <= HIT_MAX_MM:
            hit = target
    if hit is not None:
        hit["up"] = False
    hits.append((round(elapsed, 1), hit["digit"] if hit else "外れ"))
    print("    [仮EV3] 発射 {:.1f}秒 結果={}".format(elapsed, hits[-1][1]), flush=True)


def fake_ev3():
    """仮のEV3サーバ．センサ値を送り，指令を受け取ってロボットを動かす"""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", PORT))
    server.listen(1)
    conn, _ = server.accept()
    conn.settimeout(1.0)
    left = right = 0
    last_fire_id = None
    start = last = time.monotonic()
    buffer = b""
    while time.monotonic() - start < DURATION_S:
        now = time.monotonic()
        dt, last = now - last, now
        with lock:
            # 走行：左が速いと時計回りに曲がる
            v = (left + right) / 2 * WHEEL_MM_PER_DEG
            w = math.degrees((left - right) * WHEEL_MM_PER_DEG / TRACK_MM)
            robot["h"] += w * dt
            heading = math.radians(robot["h"])
            robot["x"] += v * dt * math.sin(heading)
            robot["y"] += v * dt * math.cos(heading)
            sensor = "{},{}\n".format(ultrasonic(), int(round(robot["h"])))
        try:
            conn.send(sensor.encode())
            data = conn.recv(1024)
        except socket.timeout:
            left = right = 0      # 本物と同じく，指令が届かなければ止まる
            continue
        except OSError:
            break
        if not data:
            break
        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            left, right, fire_id = map(int, line.decode().split(","))
            if last_fire_id is None:
                last_fire_id = fire_id          # 接続直後の番号は基準として覚えるだけ
            elif fire_id != last_fire_id:
                last_fire_id = fire_id
                with lock:
                    record_shot(now - start)
        time.sleep(0.02)
    conn.close()
    server.close()


def main():
    model = rc.load_model(rc.MODEL_PATH)
    if model is None:
        return
    rc.predict_digit(model, np.zeros((1, 28, 28, 1), np.float32))
    camera = SimCamera()
    worker = rc.DetectionWorker(model, camera)
    strategy = st.Strategy()
    commander = rc.Commander()

    ev3 = threading.Thread(target=fake_ev3, daemon=True)
    ev3.start()
    time.sleep(0.3)
    conn = rc.connect_ev3("127.0.0.1", PORT)

    def run():
        try:
            rc.run_session(conn, worker, strategy, commander)
        except OSError:
            pass
    threading.Thread(target=run, daemon=True).start()

    start = time.monotonic()
    while ev3.is_alive() and strategy.state != "DONE":
        time.sleep(0.2)
    time.sleep(0.5)
    worker.stop()
    conn.close()

    order = [digit for _, digit in hits]
    print("[RasPi] 経過 {:.1f}秒，最終状態 {}".format(time.monotonic() - start, strategy.state))
    print("[RasPi] 発射の記録（秒，結果）:", hits)
    print("[RasPi] ロボットの最終位置:", {k: round(v) for k, v in robot.items()})
    print("[RasPi] 結果:", "成功" if order == ["1", "2", "3"] and strategy.state == "DONE" else "失敗")


main()
