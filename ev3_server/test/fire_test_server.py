#!/usr/bin/env pybricks-micropython
# 発射用モータ（ポートD）の動作確認用サーバ（EV3側）．
# Raspberry Piから "FIRE" を受け取ったら，モータDを時計回りに1回転させて，"DONE" を返す．
# （ブロックプログラムの「Dを時計回りに開始 → 100%の速さで1回転 → Dを停止」と同じ動き）
import socket

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop, Direction

# ---- パラメータ ----
HOST = '10.42.0.175'      # EV3のIPアドレス（ディスプレイで確認する）
PORT = 9999
FIRE_PORT = Port.D        # 発射用モータのポート
FIRE_SPEED = 1000         # 発射時の速さ（度/秒）．EV3大型モータの100%はおよそ1000
FIRE_ANGLE = 360          # 1回転 = 360度

ev3 = EV3Brick()
fire_motor = Motor(FIRE_PORT, Direction.CLOCKWISE)   # CLOCKWISE = 時計回りを正とする


def fire():
    """発射用モータを時計回りに1回転させて止める．"""
    fire_motor.run_angle(FIRE_SPEED, FIRE_ANGLE, then=Stop.BRAKE, wait=True)
    fire_motor.stop()


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # 強制終了後の「使用中」エラー対策
server.bind((HOST, PORT))
server.listen(1)
print("[EV3] 接続を待っています...")
conn, addr = server.accept()
print("[EV3] 接続されました: " + str(addr))

buffer = b""
try:
    while True:
        chunk = conn.recv(1024)
        if not chunk:                       # Pi側が切断した
            print("[EV3] 切断されました")
            break
        buffer += chunk
        # 1行（\n区切り）ごとに命令を処理する
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            command = line.decode().strip()
            if command == "FIRE":
                print("[EV3] FIRE を受信．発射します")
                fire()
                conn.send(b"DONE\n")
                print("[EV3] 発射完了")
            else:
                print("[EV3] 不明な命令: " + command)
                conn.send(b"ERROR\n")
finally:
    fire_motor.stop()        # 何があっても必ず止める
    conn.close()
    server.close()
