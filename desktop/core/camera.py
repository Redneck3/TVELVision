import platform
import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from core.config import (
    DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS
)

class CameraManager:
    """Минималистичный менеджер камеры (OpenCV).
    через QTimer.
    """
    def __init__(self, index: int | None = None, status_label: QLabel | None = None):
        self.index = DEFAULT_CAMERA_INDEX if index is None else index
        self.cap = None
        self.status_label = status_label

    def open(self) -> bool:
        try:
            if platform.system() == "Windows":
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.index)

            if not self.cap or not self.cap.isOpened():
                if self.status_label:
                    self.status_label.setText("❌ Камера недоступна")
                    self.status_label.setAlignment(Qt.AlignCenter)
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
            return True
        except Exception as e:
            if self.status_label:
                self.status_label.setText(f"Ошибка запуска камеры: {e}")
            return False

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
