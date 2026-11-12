"""判断：探索 → 照準 → 接近 → 発射 → 後退 → 次の数字

センサ値と検出結果を受け取り，走行の速度と発射するかどうかを決める．
EV3にもカメラにも触れないので，作ったデータを渡して動きを試せる．

状態の流れ（狙う数字は 1 → 2 → 3 の順に進める）:
  SEARCH   : その場でゆっくり旋回して，狙う数字を探す
  AIM      : 止まったまま旋回し，狙う数字を正面に合わせる
  APPROACH : 向きを保ったまま前進し，発射距離まで近づく
  FIRE     : 止まって，正面に狙う数字があることを確かめてから1発撃つ
  FIRED    : 弾が出るまで待つ
  RETREAT  : 前進した分だけ後退して，スタート付近に戻る（的はスタート位置の方を向いているため）
  DONE     : 3に向けて撃ち終えた，または制限時間を過ぎたので止まる

向きの合わせ方:
  推論には時間がかかるので，検出結果の映像は少し古い．そこで，検出結果からは
  「どの向きに数字があるか（目標の向き）」だけを求め，実際に回る量はジャイロで測る．
  ジャイロの値は通信のたびに届くので，推論が遅くても行き過ぎにくい．
  止まった後は推論1回分以上待ってから，新しい結果で向きを確かめ直す．

モータの回る向き（左右の速度の正負）とジャイロの正負は実機で確かめていない．
逆だったら，下の GYRO_SIGN や EV3側の LEFT_DIR / RIGHT_DIR を直す．
"""
from collections import namedtuple

# ---- 順番と認識 ----
TARGET_ORDER = (1, 2, 3)      # 倒す順番
CONF_TH = 95.0                # この確信度(%)以上の検出だけを使う
CONFIRM_COUNT = 3             # 狙う数字がこの回数続けて見えたら，的を確定する
LOST_TIMEOUT_S = 2.0          # 狙う数字がこの時間見えなければ，見失ったとみなす

# ---- カメラとジャイロ ----
CAMERA_HFOV_DEG = 55.0        # カメラの水平の画角(度)．C270N のおおよその値．実測して直す
GYRO_SIGN = 1                 # 時計回り（右回り）でジャイロの角度が増えるなら 1，減るなら -1

# ---- 旋回 ----
SEARCH_TURN_SPEED = 80        # 探索中の旋回の速さ（モータの度/秒）．速いと推論が追いつかない
TURN_GAIN = 6.0               # 向きのずれ1度あたりの旋回の速さ（モータの度/秒）
TURN_MAX_SPEED = 150          # 照準中の旋回の最大の速さ
TURN_MIN_SPEED = 40           # これより遅いと摩擦で回らないので，回すときはこれ以上にする
AIM_TOLERANCE_DEG = 2.0       # 正面とみなす向きのずれ(度)
SETTLE_S = 0.8                # 止まってから向きを確かめ直すまで待つ時間．推論1回分より長くする

# ---- 接近と発射 ----
APPROACH_SPEED = 200          # 接近の速さ（モータの度/秒）
APPROACH_STEER_MAX = 60       # 接近中に向きを直すときの左右の速度差の最大
FIRE_DISTANCE_MM = 300        # 発射する距離(mm)．保留中．数字を読める距離と銃の射程から決める
DISTANCE_TOLERANCE_MM = 30    # 発射距離の許容幅(mm)
BACK_SPEED = 100              # 近づきすぎたときに下がる速さ
NO_ECHO_MM = 2550             # 超音波センサが何も検出しないときの値．これ以上は「遠い」とみなす
FIRE_WAIT_S = 1.5             # 発射の指令を出してから，次の動作を始めるまで待つ時間

# ---- 後退と終了 ----
RETREAT_SPEED = 200           # 後退の速さ（モータの度/秒）
TIME_LIMIT_S = None           # 制限時間(秒)．課題で決まっていないので None（制限なし）
MAX_DT_S = 0.2                # 1回の step() で数える経過時間の上限(秒)

# 1回分の指令．left, right は走行モータの速度（度/秒），fire は今回1発撃つかどうか
Action = namedtuple("Action", ["left", "right", "fire"])
STOP = Action(0, 0, False)


def clamp(value, low, high):
    return max(low, min(high, value))


def turn_speed(error_deg):
    """向きのずれから旋回の速さを決める．正なら時計回り（右回り）"""
    if abs(error_deg) < AIM_TOLERANCE_DEG / 2:
        return 0
    speed = clamp(TURN_GAIN * error_deg, -TURN_MAX_SPEED, TURN_MAX_SPEED)
    if abs(speed) < TURN_MIN_SPEED:
        speed = TURN_MIN_SPEED if speed > 0 else -TURN_MIN_SPEED
    return speed


def spin(speed):
    """その場で旋回する指令．正なら時計回り（左を前，右を後ろに回す）"""
    return Action(int(speed), int(-speed), False)


class Strategy:
    """状態を持ち，step() を呼ぶたびに次の指令を返す

    EV3と再接続しても作り直さない（どの数字まで撃ったかを覚えておくため）．
    """

    def __init__(self):
        self.target_index = 0            # TARGET_ORDER の何番目を狙っているか
        self.state = "SEARCH"
        self.state_since = None          # 今の状態に入った時刻
        self.start_time = None           # 最初に step() が呼ばれた時刻
        self.last_now = None
        self.last_result_id = None       # 前回使った検出結果の番号
        self.confirm = 0                 # 狙う数字が続けて見えた回数
        self.last_seen = None            # 狙う数字を最後に見た時刻
        self.target_heading = None       # 狙う数字がある向き（ジャイロの角度で表す）
        self.forward_time = 0.0          # 前進した時間（後退で戻る量に使う）
        self.check_after = None          # この時刻以降に終えた推論の結果で向きを確かめる

    @property
    def target(self):
        if self.target_index >= len(TARGET_ORDER):
            return None
        return TARGET_ORDER[self.target_index]

    def _set_state(self, state, now, reason):
        target = self.target if self.target is not None else "なし"
        print("[RasPi] 状態: {} → {}（狙う数字 {}，{}）".format(
            self.state, state, target, reason), flush=True)
        self.state = state
        self.state_since = now

    def _find_target(self, detections):
        """検出結果から，確信度が基準以上の狙う数字を1つ選ぶ（なければ None）"""
        for d in detections:      # 確信度の高い順に並んでいる
            if d.digit == self.target and d.conf >= CONF_TH:
                return d
        return None

    def step(self, now, distance_mm, gyro_angle, result):
        """次の指令（Action）を返す

        now        : 現在時刻（time.monotonic() の値）
        distance_mm: 超音波センサの距離(mm)
        gyro_angle : ジャイロの角度(度)
        result     : (結果の番号, 推論を終えた時刻, 検出結果のリスト)．
                     カメラや推論が止まっていて使えないときは None
        """
        if self.start_time is None:
            self.start_time = now
            self.state_since = now
        # 再接続などで間が空いたときに，前進や後退の量を大きく数えすぎないようにする
        dt = 0.0 if self.last_now is None else min(now - self.last_now, MAX_DT_S)
        self.last_now = now
        heading = GYRO_SIGN * gyro_angle

        if self.state == "DONE":
            return STOP
        if TIME_LIMIT_S is not None and now - self.start_time >= TIME_LIMIT_S:
            self._set_state("DONE", now, "制限時間")
            return STOP
        # 発射の待ちと後退は，カメラを使わないので検出結果がなくても続ける
        if self.state == "FIRED":
            return self._fired(now)
        if self.state == "RETREAT":
            return self._retreat(now, dt)
        if result is None:
            # 古い判断で動き続けないように止まる（状態はそのまま）
            return STOP

        result_id, result_time, detections = result
        is_new = result_id != self.last_result_id
        self.last_result_id = result_id
        found = self._find_target(detections) if is_new else None
        if found is not None:
            self.last_seen = now

        if self.state == "SEARCH":
            return self._search(now, heading, is_new, found)
        if self.state == "AIM":
            return self._aim(now, heading, result_time, is_new, found)
        if self.state == "APPROACH":
            return self._approach(now, dt, heading, distance_mm, is_new, found)
        if self.state == "FIRE":
            return self._fire(now, heading, distance_mm, result_time, is_new, found)
        return STOP

    def _lost(self, now):
        return self.last_seen is None or now - self.last_seen >= LOST_TIMEOUT_S

    def _search(self, now, heading, is_new, found):
        if is_new:
            self.confirm = self.confirm + 1 if found is not None else 0
        if self.confirm >= CONFIRM_COUNT:
            self.confirm = 0
            self.target_heading = heading + found.x_offset * CAMERA_HFOV_DEG
            self.check_after = None
            self._set_state("AIM", now, "的を確定")
            return STOP
        return spin(SEARCH_TURN_SPEED)

    def _aim(self, now, heading, result_time, is_new, found):
        if self._lost(now):
            self._set_state("SEARCH", now, "見失った")
            return STOP

        error = self.target_heading - heading
        speed = turn_speed(error)
        if speed != 0:
            # まだ目標の向きに届いていない．回っている間は確かめ直しを先送りする
            self.check_after = None
            return spin(speed)

        # 目標の向きに着いた．推論1回分以上待ってから，新しい結果で確かめ直す
        if self.check_after is None:
            self.check_after = now + SETTLE_S
            return STOP
        if now < self.check_after or result_time < self.check_after or not is_new:
            return STOP
        if found is None:
            return STOP          # 見えるまで待つ（見えないまま続けば見失ったとみなす）
        offset_deg = found.x_offset * CAMERA_HFOV_DEG
        if abs(offset_deg) <= AIM_TOLERANCE_DEG:
            self._set_state("APPROACH", now, "正面に合った")
            return STOP
        # まだずれているので，目標の向きを直してもう一度回る
        self.target_heading = heading + offset_deg
        self.check_after = None
        return STOP

    def _approach(self, now, dt, heading, distance_mm, is_new, found):
        if self._lost(now):
            self._set_state("SEARCH", now, "見失った")
            return STOP
        if found is not None:
            # 前進中も新しい結果で目標の向きを直す
            self.target_heading = heading + found.x_offset * CAMERA_HFOV_DEG

        if distance_mm < NO_ECHO_MM and abs(distance_mm - FIRE_DISTANCE_MM) <= DISTANCE_TOLERANCE_MM:
            self.check_after = now + SETTLE_S
            self._set_state("FIRE", now, "発射距離に着いた（{}mm）".format(distance_mm))
            return STOP
        if distance_mm < FIRE_DISTANCE_MM - DISTANCE_TOLERANCE_MM:
            self.forward_time -= dt * BACK_SPEED / APPROACH_SPEED
            return Action(-BACK_SPEED, -BACK_SPEED, False)

        steer = clamp(TURN_GAIN * (self.target_heading - heading), -APPROACH_STEER_MAX, APPROACH_STEER_MAX)
        self.forward_time += dt
        return Action(int(APPROACH_SPEED + steer), int(APPROACH_SPEED - steer), False)

    def _fire(self, now, heading, distance_mm, result_time, is_new, found):
        # 止まってから推論1回分以上待ち，その後の新しい結果で確かめてから撃つ
        if now < self.check_after or result_time < self.check_after or not is_new:
            if self._lost(now):
                self._set_state("SEARCH", now, "見失った")
            return STOP
        if found is None:
            if self._lost(now):
                self._set_state("SEARCH", now, "見失った")
            return STOP
        offset_deg = found.x_offset * CAMERA_HFOV_DEG
        if abs(offset_deg) > AIM_TOLERANCE_DEG:
            self.target_heading = heading + offset_deg
            self.check_after = None
            self._set_state("AIM", now, "向きがずれた（{:.1f}度）".format(offset_deg))
            return STOP
        if distance_mm >= NO_ECHO_MM or abs(distance_mm - FIRE_DISTANCE_MM) > DISTANCE_TOLERANCE_MM:
            self._set_state("APPROACH", now, "距離がずれた（{}mm）".format(distance_mm))
            return STOP
        self._set_state("FIRED", now, "発射")
        return Action(0, 0, True)

    def _fired(self, now):
        if now - self.state_since >= FIRE_WAIT_S:
            self._set_state("RETREAT", now, "発射を終えた")
        return STOP

    def _retreat(self, now, dt):
        if self.forward_time > 0:
            # forward_time は「接近の速さで前進した時間」なので，速さの比で換算して減らす
            self.forward_time -= dt * RETREAT_SPEED / APPROACH_SPEED
            return Action(-RETREAT_SPEED, -RETREAT_SPEED, False)
        self.forward_time = 0.0
        self.target_index += 1
        self.last_seen = None
        self.confirm = 0
        if self.target is None:
            self._set_state("DONE", now, "3発撃ち終えた")
        else:
            self._set_state("SEARCH", now, "次の数字へ")
        return STOP
