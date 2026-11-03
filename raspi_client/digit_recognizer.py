import cv2
import numpy as np
import tensorflow as tf


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
            digit = thresh[y1:y2, x1:x2]

            # 縦横比を保ったまま20x20に収め、28x28の中央に置く
            scale = 20.0 / max(x2 - x1, y2 - y1)
            w = max(1, int(round((x2 - x1) * scale)))
            h = max(1, int(round((y2 - y1) * scale)))
            digit = cv2.resize(digit, (w, h), interpolation=cv2.INTER_AREA)
            ox = (28 - w) // 2
            oy = (28 - h) // 2
            resized[oy:oy + h, ox:ox + w] = digit

    # モデルに入力するための形状変換 (1, 28, 28, 1) と正規化
    input_data = resized.reshape(1, 28, 28, 1).astype(np.float32) / 255.0

    return input_data, resized


def predict_digit(model, input_data):
    """数字を予測する"""
    # model.predict は1枚ずつ呼ぶと遅いので、直接呼び出す
    prediction = model(input_data, training=False).numpy()[0]
    predicted_num = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100  # 確信度(%)
    return predicted_num, confidence
