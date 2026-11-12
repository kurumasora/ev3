from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf

# ---- 画面全体からの数字探し（find_digits）のパラメータ．実物を映して調整する ----
FIND_MIN_STD = 5              # 明暗の差（標準偏差）がこれ未満なら，何も映っていないとみなす．遠くの小さな数字では差が小さいので低めにする
FIND_MIN_HEIGHT_RATIO = 0.05  # 候補の高さの最小（画面の高さに対する割合）．小さなゴミを除く
FIND_MAX_HEIGHT_RATIO = 0.9   # 候補の高さの最大．画面いっぱいの影などを除く
FIND_MIN_ASPECT = 1.0         # 候補の 高さ÷幅 の最小．数字は縦長なので，横長のもの（パネルの縁など）を除く
FIND_MAX_ASPECT = 8.0         # 候補の 高さ÷幅 の最大．1 は細長いので広めにとる
FIND_MAX_CANDIDATES = 6       # 推論する候補の最大数（大きい順）．推論は候補1個につき約65msかかる

# find_digits が返す検出結果
#   digit    : 予測した数字
#   conf     : 確信度(%)
#   x_offset : 画面中央からの横のずれ（画面の幅に対する割合，-0.5〜0.5．右が正）
#   height   : 数字の高さ（画面の高さに対する割合）
#   box      : 画面上の範囲 (x, y, 幅, 高さ)．確認ツールでの表示用
Detection = namedtuple("Detection", ["digit", "conf", "x_offset", "height", "box"])


def load_model(model_path='mnist_model.keras'):
    """学習済みモデルを読み込む"""
    try:
        model = tf.keras.models.load_model(model_path)
        print("モデルを読み込みました。")
        return model
    except:
        print(f"エラー: '{model_path}' が見つかりません。先に train.py を実行してください。")
        return None


def preprocess_image(roi, min_area_ratio=0.002):
    """画像を前処理してモデル入力形式に変換

    MNISTと同じ形（黒背景に白文字、20x20に収めて28x28の中央に配置）にそろえる。
    文字が見つからない場合、resized は真っ黒になる。
    """
    # グレースケール変換 + ノイズを減らすためのぼかし
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    resized = np.zeros((28, 28), np.uint8)

    # 明暗の差がほとんどない = 何も書かれていない
    if blur.std() >= 15:
        # 二値化 (大津の方法で閾値を自動決定。照明が変わっても追従できる)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # 小さなゴミを除いた輪郭をまとめて、文字のある範囲を求める
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = min_area_ratio * thresh.shape[0] * thresh.shape[1]
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]

        if boxes:
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            resized = fit_to_mnist(thresh[y1:y2, x1:x2])

    # モデルに入力するための形状変換 (1, 28, 28, 1) と正規化
    input_data = resized.reshape(1, 28, 28, 1).astype(np.float32) / 255.0

    return input_data, resized


def fit_to_mnist(digit):
    """白黒の文字画像を，縦横比を保ったまま20x20に収め，28x28の中央に置く"""
    resized = np.zeros((28, 28), np.uint8)
    h0, w0 = digit.shape[:2]
    scale = 20.0 / max(w0, h0)
    w = max(1, int(round(w0 * scale)))
    h = max(1, int(round(h0 * scale)))
    digit = cv2.resize(digit, (w, h), interpolation=cv2.INTER_AREA)
    ox = (28 - w) // 2
    oy = (28 - h) // 2
    resized[oy:oy + h, ox:ox + w] = digit
    return resized


def find_digits(model, frame):
    """画面全体から数字の候補を探し，候補ごとに推論した結果（Detection のリスト）を返す

    preprocess_image は認識エリアの中の輪郭をすべてまとめて1文字として扱うが，
    こちらは輪郭を1つずつ別の候補として扱うので，数字の位置が分かる．
    数字らしくない形（小さすぎる，横長，画面の端に接している）は推論する前に捨てる．
    結果は確信度の高い順に並べる．数字が見つからなければ空のリストを返す．
    """
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if blur.std() < FIND_MIN_STD:
        return []

    # 暗い文字を白，明るい背景を黒にする（MNISTと同じ向き）
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 背景がパネルより暗いと，背景・パネル・数字が入れ子になる（数字はパネルの穴の中の白い塊）．
    # 入れ子の深さが偶数の輪郭（白い塊の外周）を全部候補にし，奇数（穴の縁）は除く
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    parents = hierarchy[0][:, 3]

    candidates = []
    for i, c in enumerate(contours):
        depth = 0
        parent = parents[i]
        while parent != -1:
            depth += 1
            parent = parents[parent]
        if depth % 2 == 1:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (FIND_MIN_HEIGHT_RATIO * frame_h <= h <= FIND_MAX_HEIGHT_RATIO * frame_h):
            continue
        if not (FIND_MIN_ASPECT <= h / w <= FIND_MAX_ASPECT):
            continue
        # 画面の端に接しているものは，背景や見切れた物の可能性が高い
        if x <= 0 or y <= 0 or x + w >= frame_w or y + h >= frame_h:
            continue
        candidates.append((h, c, (x, y, w, h)))

    # 大きい順に，推論する数を制限する
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:FIND_MAX_CANDIDATES]
    if not candidates:
        return []

    images = []
    for _, c, (x, y, w, h) in candidates:
        # 範囲内に入り込んだ別の輪郭を除くため，その輪郭の内側だけを残す
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED, offset=(-x, -y))
        digit = cv2.bitwise_and(thresh[y:y + h, x:x + w], mask)
        images.append(fit_to_mnist(digit))

    # 候補をまとめて1回で推論する（1個ずつ推論するより速い）
    batch = np.stack(images).reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    predictions = model(batch, training=False).numpy()

    detections = []
    for (_, _, (x, y, w, h)), prediction in zip(candidates, predictions):
        detections.append(Detection(
            digit=int(np.argmax(prediction)),
            conf=float(np.max(prediction)) * 100,
            x_offset=(x + w / 2 - frame_w / 2) / frame_w,
            height=h / frame_h,
            box=(x, y, w, h),
        ))
    detections.sort(key=lambda d: d.conf, reverse=True)
    return detections


def predict_digit(model, input_data):
    """数字を予測する"""
    # model.predict は1枚ずつ呼ぶと遅いので、直接呼び出す
    prediction = model(input_data, training=False).numpy()[0]
    predicted_num = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100  # 確信度(%)
    return predicted_num, confidence
