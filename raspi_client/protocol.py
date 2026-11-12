"""EV3との通信形式の組み立てと解析

通信形式（CLAUDE.md の「通信形式」と同じ．変えるときは ev3_server/motor_sensor_server.py も同時に直す）:
  EV3 → Pi : "distance_mm,gyro_angle\\n"
      distance_mm : 超音波センサの距離(mm)
      gyro_angle  : ジャイロの角度(度)．時計回りが正
  Pi → EV3 : "left_speed,right_speed,fire_id\\n"
      left_speed, right_speed : 左右の走行モータの速度(度/秒)．整数
      fire_id : 発射番号．前回より変わったときだけ，EV3は1発撃つ
                同じ行が何回届いても二重に撃たないようにするため

EV3がセンサ値を1行送るたびに，Piが指令を1行返す．
"""


def parse_sensor_line(line):
    """ "distance_mm,gyro_angle" を (距離, 角度) にする．壊れた行なら None を返す"""
    try:
        distance_mm, gyro_angle = map(int, line.split(","))
    except ValueError:
        return None
    return distance_mm, gyro_angle


def format_command(left_speed, right_speed, fire_id):
    """指令を1行のバイト列にする"""
    return "{},{},{}\n".format(int(left_speed), int(right_speed), int(fire_id)).encode("utf-8")


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
