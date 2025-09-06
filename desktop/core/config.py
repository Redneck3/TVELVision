import os

# Папка для изображений брака
DEFECT_FOLDER = os.path.join(os.getcwd(), "defect_images")

# Путь к модели YOLO
YOLO_MODEL_PATH = os.path.join(
    os.getcwd(),
    "runs", "detect", "train3", "weights", "best.pt"
)

# Параметры камеры
DEFAULT_CAMERA_INDEX = 2
FRAME_WIDTH = 800
FRAME_HEIGHT = 640
FRAME_FPS = 60

# Интервалы
SPAWN_INTERVAL_MS = 800
UI_TIMER_MS = 30
CAM_TIMER_MS = 30
