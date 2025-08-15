import os
import numpy as np
import cv2
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from pathlib import Path

#TODO: Кофликтные ситуации keras и tf

data = []
labels = []

IMG_SIZE = 64
for label, category in enumerate(["not_tablet", "tablet"]):
    folder = f'E:/Code/TVELVision/web/ai/dataset/{category}'
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        data.append(img)
        labels.append(label)

X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


APP_DIR = Path(__file__).parent
model.save( APP_DIR / "h5"/ "tablet_detector.h5", save_format="h5")