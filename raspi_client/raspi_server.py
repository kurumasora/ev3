# raspi_server_stream_nogui.py
import socket
import time
import cv2
from camera import load_model, preprocess_image, predict_digit

HOST = "192.168.11.11"
PORT = 5000

CAMERA_ID = 0

# 負荷が高いなら 800→400（まずここ）
ROI_SIZE = 400

CONF_TH = 95.0
TARGET_DIGITS = {1, 2, 3}

NEED_GO_FRAMES = 3
NEED_STOP_FRAMES = 3

# EV3側のブロッキングrecv対策：同じ状態でも定期送信
KEEPALIVE_S = 0.5

# 推論を間引く（3なら約1/3の負荷）
INFER_EVERY = 3

# ログ頻度
LOG_EVERY_N = 10

def accept_ev3(server_sock):
    print(f"[RasPi] Listening on {HOST}:{PORT}", flush=True)
    conn, addr = server_sock.accept()
    print(f"[RasPi] EV3 connected from {addr}", flush=True)
    try:
        conn.settimeout(0.2)
    except Exception:
        pass
    return conn

def main():
    model = load_model("mnist_model.keras")
    if model is None:
        print("[RasPi] モデル読み込み失敗", flush=True)
        return

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[RasPi] カメラ {CAMERA_ID} を開けませんでした", flush=True)
        return

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    conn = accept_ev3(server)

    state = "STOP"
    last_send_t = 0.0
    seen_cnt = 0
    not_seen_cnt = 0

    frame_i = 0

    # 推論結果を保持（間引きフレームでも直近値で判定できる）
    pred, conf, visible = -1, 0.0, False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[RasPi] frame read failed", flush=True)
                break
            frame_i += 1

            # 推論はINFER_EVERYフレームに1回だけ
            if frame_i % INFER_EVERY == 0:
                h, w, _ = frame.shape
                x1 = (w - ROI_SIZE) // 2
                y1 = (h - ROI_SIZE) // 2
                x2 = x1 + ROI_SIZE
                y2 = y1 + ROI_SIZE
                roi = frame[y1:y2, x1:x2]

                try:
                    input_data, _ = preprocess_image(roi)
                    pred, conf = predict_digit(model, input_data)
                except Exception as e:
                    print("[RasPi] predict error:", e, flush=True)
                    pred, conf = -1, 0.0

                visible = (conf >= CONF_TH) and (pred in TARGET_DIGITS)

            # デバウンス（visibleは直近推論の結果を使う）
            if visible:
                seen_cnt += 1
                not_seen_cnt = 0
            else:
                not_seen_cnt += 1
                seen_cnt = 0

            new_state = state
            if state == "STOP" and seen_cnt >= NEED_GO_FRAMES:
                new_state = "GO"
            elif state == "GO" and not_seen_cnt >= NEED_STOP_FRAMES:
                new_state = "STOP"

            if frame_i % LOG_EVERY_N == 0:
                print(
                    f"[RasPi] pred={pred} conf={conf:.1f}% visible={visible} "
                    f"seen={seen_cnt} not_seen={not_seen_cnt} state={state}->{new_state}",
                    flush=True
                )

            now = time.time()

            # 送信（状態変化 or keepalive）
            if (new_state != state) or (now - last_send_t >= KEEPALIVE_S):
                try:
                    conn.sendall((new_state + "\n").encode("utf-8"))
                    last_send_t = now
                    print(f"[RasPi] SENT: {new_state} (pred={pred}, conf={conf:.1f}%)", flush=True)
                except Exception as e:
                    print("[RasPi] send error:", e, flush=True)

                    # 切断復帰
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = accept_ev3(server)

                    # 再接続直後は安全にSTOPへ
                    state = "STOP"
                    last_send_t = 0.0
                    seen_cnt = 0
                    not_seen_cnt = 0
                    pred, conf, visible = -1, 0.0, False
                    continue

            state = new_state

    finally:
        try:
            conn.sendall(b"STOP\n")
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            server.close()
        except Exception:
            pass
        cap.release()

if __name__ == "__main__":
    main()
