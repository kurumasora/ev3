# 発射用モータの動作確認用クライアント（Raspberry Pi側）．
# EV3へ "FIRE" を送り，発射が終わって "DONE" が返ってくるのを待つ．
# 使い方：  python fire_test_client.py        （1発）
#           python fire_test_client.py 3      （3発．1発ごとに DONE を待つ）
import socket
import sys

# ---- パラメータ ----
EV3_IP = '10.42.0.175'    # EV3のIPアドレス（ディスプレイで確認する）
PORT = 9999
REPLY_TIMEOUT = 10.0      # DONE を待つ最大秒数
INTERVAL_NOTE = "1発ごとに発射完了を待ちます"


def wait_for_reply(sock):
    """EV3から1行（\\n区切り）の返事を受け取る．"""
    buffer = b""
    while b"\n" not in buffer:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("EV3との接続が切れました")
        buffer += chunk
    return buffer.split(b"\n", 1)[0].decode().strip()


def main():
    shots = 1
    if len(sys.argv) >= 2:
        shots = int(sys.argv[1])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(REPLY_TIMEOUT)
    print("[RasPi] EV3 (" + EV3_IP + ") に接続します...")
    sock.connect((EV3_IP, PORT))
    print("[RasPi] 接続しました．" + INTERVAL_NOTE)

    try:
        for i in range(shots):
            print("[RasPi] " + str(i + 1) + "発目: FIRE を送信")
            sock.send(b"FIRE\n")
            reply = wait_for_reply(sock)
            print("[RasPi] EV3の返事: " + reply)
            if reply != "DONE":
                print("[RasPi] 発射に失敗したので中止します")
                break
    except KeyboardInterrupt:
        print("[RasPi] 中断しました")
    finally:
        sock.close()


main()
