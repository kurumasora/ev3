# 推論にかかる時間を測る（モータもカメラも使わない）．
# 前処理（400px四方の画像）と，数字1つ分の推論の1回あたりの時間を表示する．
# 使い方：  cd raspi_client && uv run python test/bench_inference.py
import os
import sys
import time

import numpy as np

# 親フォルダ（raspi_client/）のファイルを import し，モデルを相対パスで読み込めるようにする
CLIENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLIENT_DIR)
os.chdir(CLIENT_DIR)

from digit_recognizer import load_model, predict_digit, preprocess_image  # noqa: E402

# ---- パラメータ ----
MODEL_PATH = "mnist_model.keras"
ROI_SIZE = 400        # 前処理する画像の一辺(px)
REPEAT = 50           # 何回測って平均するか


def main():
    model = load_model(MODEL_PATH)
    if model is None:
        return
    blank = np.zeros((1, 28, 28, 1), np.float32)
    predict_digit(model, blank)   # 最初の推論は遅いので，測る前に1回済ませる

    roi = (np.random.rand(ROI_SIZE, ROI_SIZE, 3) * 255).astype(np.uint8)
    t = time.perf_counter()
    for _ in range(REPEAT):
        preprocess_image(roi)
    pre_ms = (time.perf_counter() - t) / REPEAT * 1000

    t = time.perf_counter()
    for _ in range(REPEAT):
        predict_digit(model, blank)
    infer_ms = (time.perf_counter() - t) / REPEAT * 1000

    print("[RasPi] 前処理 {:.1f}ms，推論 {:.1f}ms（1回あたり，{}回の平均）".format(pre_ms, infer_ms, REPEAT))


main()
