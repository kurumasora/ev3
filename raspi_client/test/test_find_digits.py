# find_digits（画面全体から数字を探す処理）の確認（モータもカメラも使わない）．
# 灰色の背景に白いパネルと黒い数字を描いた合成画像で，数字・確信度・位置が正しく出るかを見る．
# 本物のカメラと印刷の数字での確認ではないので，実物では recognition_viewer.py なども使って確かめる．
# 使い方：  cd raspi_client && uv run python test/test_find_digits.py
import os
import sys
import time

import cv2
import numpy as np

# 親フォルダ（raspi_client/）のファイルを import し，モデルを相対パスで読み込めるようにする
CLIENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLIENT_DIR)
os.chdir(CLIENT_DIR)

from digit_recognizer import find_digits, load_model, predict_digit  # noqa: E402

# ---- パラメータ ----
MODEL_PATH = "mnist_model.keras"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BACKGROUND = 200      # 背景の明るさ（パネルより少し暗くする）
NOISE = 6             # 背景のざらつき（標準偏差）

# 確認する場面：(名前, [(数字, 中心のx座標, 大きさ), ...], 期待する数字の組)
CASES = [
    ("中央に1", [("1", 320, 120)], {1}),
    ("中央に2", [("2", 320, 120)], {2}),
    ("中央に3", [("3", 320, 120)], {3}),
    ("左に小さな2", [("2", 150, 80)], {2}),
    ("右の遠くに3", [("3", 500, 60)], {3}),
    ("1,2,3が同時", [("1", 120, 100), ("2", 320, 100), ("3", 520, 100)], {1, 2, 3}),
    ("何もなし", [], set()),
]

rng = np.random.default_rng(0)


def make_scene(items):
    """背景に，白いパネルとその上の黒い数字を描いた画像を作る"""
    img = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BACKGROUND, np.uint8)
    img = np.clip(img + rng.normal(0, NOISE, img.shape), 0, 255).astype(np.uint8)
    cy = FRAME_HEIGHT // 2
    for text, cx, size in items:
        pw, ph = int(size * 0.9), int(size * 1.4)
        cv2.rectangle(img, (cx - pw // 2, cy - ph // 2), (cx + pw // 2, cy + ph // 2), (250, 250, 250), -1)
        scale, thick = size / 22.0, max(2, int(size / 9))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick)
    return img


def main():
    model = load_model(MODEL_PATH)
    if model is None:
        return
    predict_digit(model, np.zeros((1, 28, 28, 1), np.float32))   # 最初の推論は遅いので先に済ませる

    passed = 0
    for name, items, expected in CASES:
        t = time.perf_counter()
        detections = find_digits(model, make_scene(items))
        ms = (time.perf_counter() - t) * 1000
        found = {d.digit for d in detections if d.conf >= 95.0}
        ok = found == expected
        passed += ok
        text = ", ".join("{}:{:.0f}% 位置{:+.2f} 高さ{:.2f}".format(d.digit, d.conf, d.x_offset, d.height)
                         for d in detections) or "なし"
        print("[RasPi] {} {:<12} {:4.0f}ms  {}".format("OK  " if ok else "NG  ", name, ms, text))
    print("[RasPi] {}/{} 件が期待どおり".format(passed, len(CASES)))


main()
