from ultralytics import YOLO

from core.config import YOLO_MODEL_PATH

class YoloDetector:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.model = YOLO(self.model_path)

    def predict(self, frame, imgsz: int = 640, conf: float = 0.085):
        """Возвращает results ultralytics; на вход — BGR кадр (cv2)."""
        return self.model(frame, imgsz=imgsz, conf=conf)
