import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === Параметры ===
IMG_SIZE = 64
DATASET_DIR = Path("E:/Code/TVELVision/ai/dataset")
YOLO_LABELS_DIR = DATASET_DIR / "labels"
YOLO_LABELS_DIR.mkdir(exist_ok=True, parents=True)

def enhance_dataset_with_lighting_variations(manual_roi=True):
    """Создание датасета с вариациями освещения и сохранением YOLO-разметки"""
    data = []
    labels = []
    
    for label, category in enumerate(["not_tablet", "tablet"]):
        folder = DATASET_DIR / category
        for file in os.listdir(folder):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = str(folder / file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # --- Выбор ROI ---
                if manual_roi:
                    roi = cv2.selectROI(f"Выделите область для {category}", img, fromCenter=False, showCrosshair=True)
                    if roi == (0, 0, 0, 0):  
                        print(f"Пропущено: {img_path}")
                        continue
                    x, y, w, h = roi
                    cropped = img[y:y+h, x:x+w]

                    # Сохраняем YOLO-аннотацию
                    H, W = img.shape[:2]
                    x_center = (x + w/2) / W
                    y_center = (y + h/2) / H
                    w_norm = w / W
                    h_norm = h / H
                    yolo_line = f"{label} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

                    txt_path = YOLO_LABELS_DIR / (Path(file).stem + ".txt")
                    with open(txt_path, "w") as f:
                        f.write(yolo_line)
                else:
                    cropped = img  # берём всё изображение, если ROI не нужен

                # --- Подготовка изображения ---
                img_resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                data.append(img_resized)
                labels.append(label)

                # --- Аугментации по освещению ---
                for alpha in [0.6, 0.8, 1.0, 1.2, 1.4]:
                    for beta in [-30, 0, 30]:
                        adjusted = cv2.convertScaleAbs(img_resized, alpha=alpha, beta=beta)
                        data.append(adjusted)
                        labels.append(label)

    cv2.destroyAllWindows()
    return np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0, np.array(labels)

# === Создание датасета ===
print("Создание датасета с вариациями освещения...")
X, y = enhance_dataset_with_lighting_variations(manual_roi=True)
print(f"Создано {len(X)} изображений")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Модель CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# === Обучение ===
print("Начало обучения...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# === Сохранение модели ===
APP_DIR = Path(__file__).parent
save_path = APP_DIR / "ai" / "h5" / "tablet_detector_enhanced.keras"
save_path.parent.mkdir(parents=True, exist_ok=True)
model.save(save_path)

# === Оценка ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

print("✅ Модель обучена и сохранена!")
print(f"YOLO разметка сохранена в: {YOLO_LABELS_DIR}")
