import numpy as np
import cv2
from keras.models import load_model


model = load_model("E:/Code/TVELVision/h5/tablet_detector.h5")

def is_tablet(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, 64, 64, 1)

    prediction = float(model.predict(img, verbose=0)[0][0])

    print("=" * 40)
    print(f"📄 Файл: {image_path}")
    print(f"Уверенность модели: {prediction:.4f}")
    if prediction > 0.5:
        print("Обнаружена таблетка")
    else:
        print("Таблетка не обнаружена")
    print("=" * 40)
    return prediction > 0.5

print(is_tablet("E:/Code/TVELVision/web/ai/test/test_1.jpg"))
print(is_tablet("E:/Code/TVELVision/web/ai/test/test_2.jpg"))
print(is_tablet("E:/Code/TVELVision/web/ai/test/test_3.jpg"))
print(is_tablet("E:/Code/TVELVision/web/ai/test/test_4.jpg"))