"""カメラからのフレーム取得と，認識エリアの計算"""
import threading
import time

import cv2


class LatestFrameReader:
    """別スレッドでカメラを読み続け，常に最新の1フレームだけを保持する

    推論が撮影より遅くても，古いフレームがたまって映像が遅れることがない．
    """

    def __init__(self, camera_id, width, height):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = self.cap.isOpened()
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1

    def read(self):
        with self.lock:
            return self.seq, self.frame

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()


def center_roi(frame, roi_size):
    """フレーム中央の正方形の認識エリアを (x1, y1, x2, y2) で返す"""
    h, w = frame.shape[:2]
    s = min(roi_size, w, h)
    x1 = (w - s) // 2
    y1 = (h - s) // 2
    return x1, y1, x1 + s, y1 + s
