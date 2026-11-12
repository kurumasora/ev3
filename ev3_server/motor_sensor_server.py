#!/usr/bin/env pybricks-micropython
# ============================================================================
# motor_sensor_server.py  ―  EV3側のTCPサーバ（本番用．このファイル1つだけをEV3に送る）
#
# 役割：判断はしない．次の2つだけを行う．
#   1. センサ値（超音波センサの距離など）をRaspberry Piへ送る．
#   2. Raspberry Piから届いた指令（走行・発射）を実行する．
#
# 通信形式（1行 = 1メッセージ，改行 "\n" で区切る）
#   EV3 → Pi ： distance_mm,gyro_angle
#                 距離(mm)，ジャイロの角度(度)．ジャイロを使わないときは角度は常に0．
#   Pi → EV3 ： left_speed,right_speed,fire_id
#                 左右のモータ速度(度/秒，整数)，発射番号．
#                 発射番号は省略してもよい（その場合は発射しない）．
#                 発射番号が前回と変わったときだけ，1発発射する．
#                 Pi側は，撃つたびに発射番号を1ずつ増やす（最初は0）．
#
# 安全のための動作
#   - 指令が COMMAND_TIMEOUT_MS ミリ秒届かないと，走行モータを止める．
#   - 切断・通信エラーのときは，全モータを止めて，接続待ちに戻る．
#   - 接続し直したとき，最初に届いた発射番号は「基準」として覚えるだけで，発射しない．
#
# 起動方法
#   - 開発中：Raspberry Piから転送して，brickrun で起動する．
#   - 本番  ：EV3本体のファイルブラウザからこのファイルを選んで起動する（chmod +x が必要）．
# ============================================================================
import socket

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, UltrasonicSensor, GyroSensor
from pybricks.parameters import Port, Stop, Direction
from pybricks.tools import wait, StopWatch

# ---------------------------------------------------------------------------
# 定数（機体に合わせてここだけを書き換える）
# ---------------------------------------------------------------------------
HOST = '0.0.0.0'                  # すべての接続を受け付ける（EV3のIPが変わっても書き換え不要）
PORT = 9999                       # 通信ポート

LEFT_MOTOR_PORT = Port.A          # 左車輪のモータ
RIGHT_MOTOR_PORT = Port.B         # 右車輪のモータ
FIRE_MOTOR_PORT = Port.D          # 発射用モータ
DISTANCE_SENSOR_PORT = Port.S1    # 超音波センサ
GYRO_SENSOR_PORT = Port.S2        # ジャイロセンサ

USE_GYRO = False                  # ジャイロを使うかどうか．使わないときは False（未接続でも起動できる）
MAX_DRIVE_SPEED = 1000            # 走行速度の上限（度/秒）．これを超える指令は丸める

FIRE_SPEED = 1000                 # 発射時のモータの速さ（度/秒）
FIRE_ANGLE = 360                  # 1発分の回転角（度）．360 = 1回転
FIRE_COOLDOWN_MS = 1500           # 発射が終わるまでの待ち時間（ミリ秒）．この間は次の発射指令を無視する

COMMAND_TIMEOUT_MS = 1000         # この時間，指令が届かなかったら走行モータを止める（ミリ秒）
LOOP_WAIT_MS = 20                 # 1ループごとの待ち時間（ミリ秒）

# 切断を意味するエラー番号（EPIPE, EBADF, ECONNRESET, ENOTCONN）．
# これ以外の受信エラーは「タイムアウト」として扱う．
DISCONNECT_ERRNOS = (32, 9, 104, 107)

# ---------------------------------------------------------------------------
# 初期化
# ---------------------------------------------------------------------------
ev3 = EV3Brick()
left_motor = Motor(LEFT_MOTOR_PORT)
right_motor = Motor(RIGHT_MOTOR_PORT)
fire_motor = Motor(FIRE_MOTOR_PORT, Direction.CLOCKWISE)   # 時計回りを正とする
distance_sensor = UltrasonicSensor(DISTANCE_SENSOR_PORT)

gyro_sensor = None
if USE_GYRO:
    gyro_sensor = GyroSensor(GYRO_SENSOR_PORT)
    gyro_sensor.reset_angle(0)

fire_timer = StopWatch()          # 前回の発射からの経過時間を測る
fire_timer.pause()
fire_started = False              # 一度でも発射したか


# ---------------------------------------------------------------------------
# 関数
# ---------------------------------------------------------------------------
def show_status(message):
    """EV3の画面に状態を表示する（本番はPCなしで動かすため，画面で確認できるようにする）．"""
    ev3.screen.clear()
    ev3.screen.print(message)
    print("[EV3] " + message)


def read_sensors():
    """センサを読んで，Raspberry Piへ送る1行（改行つき）を返す．"""
    distance = distance_sensor.distance()       # 距離(mm)
    angle = 0
    if gyro_sensor is not None:
        angle = gyro_sensor.angle()             # 角度(度)
    return str(distance) + "," + str(angle) + "\n"


def parse_command(line):
    """届いた1行を解析して (左速度, 右速度, 発射番号) を返す．
    発射番号が省略されているときは None．形式が壊れているときは全体で None を返す．"""
    parts = line.strip().split(",")
    if len(parts) < 2:
        return None
    try:
        left = int(parts[0])
        right = int(parts[1])
        fire_id = None
        if len(parts) >= 3 and parts[2].strip() != "":
            fire_id = int(parts[2])
    except ValueError:
        return None
    return (left, right, fire_id)


def clamp(value, limit):
    """値を -limit 〜 +limit の範囲に収める．"""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def drive(left_speed, right_speed):
    """走行モータを回す（速度の単位は度/秒）．0なら止める．"""
    left_speed = clamp(left_speed, MAX_DRIVE_SPEED)
    right_speed = clamp(right_speed, MAX_DRIVE_SPEED)
    if left_speed == 0:
        left_motor.brake()
    else:
        left_motor.run(left_speed)
    if right_speed == 0:
        right_motor.brake()
    else:
        right_motor.run(right_speed)


def stop_drive():
    """走行モータだけを止める（発射中の発射モータは止めない）．"""
    left_motor.brake()
    right_motor.brake()


def fire():
    """発射モータを1発分回し始める．完了は待たずに，すぐ通信へ戻る．
    前回の発射が終わっていないときは何もしない（二重に撃たない）．
    戻り値：実際に発射を始めたら True．"""
    global fire_started
    if fire_started and fire_timer.time() < FIRE_COOLDOWN_MS:
        return False
    fire_motor.run_angle(FIRE_SPEED, FIRE_ANGLE, then=Stop.BRAKE, wait=False)
    fire_timer.reset()
    fire_timer.resume()
    fire_started = True
    return True


def stop_all():
    """全モータを止める（切断・エラー・終了のとき）．"""
    left_motor.brake()
    right_motor.brake()
    fire_motor.stop()


def handle_command(command, last_fire_id):
    """解析済みの指令を実行する．更新後の「前回の発射番号」を返す．"""
    left, right, fire_id = command
    drive(left, right)
    if fire_id is None:
        return last_fire_id
    if last_fire_id is None:
        # 接続後，最初に届いた発射番号は基準として覚えるだけ（再接続で勝手に撃たない）
        return fire_id
    if fire_id != last_fire_id:
        if fire():
            print("[EV3] 発射！ 発射番号=" + str(fire_id))
            return fire_id
        # 前の発射がまだ終わっていない．番号は更新せず，次の指令でもう一度試す（撃ち漏らさない）
    return last_fire_id


def serve_one_connection(conn):
    """1つの接続を処理する．切断かエラーになったら戻る．
    流れ：センサ値を送る → 指令を受け取る → 実行する．"""
    conn.settimeout(COMMAND_TIMEOUT_MS / 1000)
    buffer = b""              # 改行までためておく受信バッファ（データが途中で分かれて届くことがあるため）
    last_fire_id = None       # 前回の発射番号

    while True:
        # --- センサ値を送る ---
        conn.send(read_sensors().encode())

        # --- 指令を受け取る ---
        try:
            chunk = conn.recv(1024)
        except OSError as error:
            code = error.args[0] if error.args else None
            if code in DISCONNECT_ERRNOS:
                return                          # 切断された
            # タイムアウト：指令が届かないので走行モータを止める．
            stop_drive()
            print("[EV3] 指令が届きません．走行モータを停止")
            continue
        if not chunk:
            return                              # 相手が接続を閉じた
        buffer += chunk

        # --- 1行ずつ処理する ---
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            command = parse_command(line.decode())
            if command is None:
                print("[EV3] 壊れた指令を無視: " + line.decode())
                continue
            last_fire_id = handle_command(command, last_fire_id)

        wait(LOOP_WAIT_MS)


# ---------------------------------------------------------------------------
# メイン：接続を待つ → 接続されたら処理する → 切れたら接続待ちに戻る
# ---------------------------------------------------------------------------
def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # 強制終了後の「使用中」エラー対策
    server.bind((HOST, PORT))
    server.listen(1)
    try:
        while True:
            show_status("Waiting...")
            conn, addr = server.accept()
            show_status("Connected")
            ev3.speaker.beep()                   # 接続できたことを音で知らせる
            try:
                serve_one_connection(conn)
            except OSError as error:
                print("[EV3] 通信エラー: " + str(error))
            finally:
                stop_all()                       # 何があっても全モータを止める
                conn.close()
            show_status("Disconnected")
    finally:
        stop_all()
        server.close()


main()
