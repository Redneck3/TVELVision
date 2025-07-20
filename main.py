import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Глобальные переменные
cap = None
running = False

def detect_defect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_defect = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            has_defect = True

    return frame, has_defect

def update_frame():
    global running, cap
    if running and cap is not None:
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        processed_frame, defect = detect_defect(frame)

        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if defect:
            status_label.config(text="⚠️ Дефект обнаружен!", fg="red")
        else:
            status_label.config(text="✅ Дефектов не обнаружено", fg="green")

        root.after(30, update_frame)

def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()
        status_label.config(text="Камера запущена", fg="black")

def stop_camera():
    global cap, running
    if running:
        running = False
        if cap is not None:
            cap.release()
            cap = None
        status_label.config(text="Камера остановлена", fg="gray")
        video_label.config(image="")

# GUI
root = tk.Tk()
root.title("Контроль качества таблеток ТВЭЛ")
root.geometry("800x600")

video_label = tk.Label(root)
video_label.pack()

status_label = tk.Label(root, text="Нажмите 'Старт' для запуска камеры", font=("Arial", 16))
status_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="▶ Старт", command=start_camera, bg="green", fg="white", width=10)
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(button_frame, text="⏹ Стоп", command=stop_camera, bg="red", fg="white", width=10)
stop_button.grid(row=0, column=1, padx=10)

root.mainloop()

# Очистка
if cap:
    cap.release()
cv2.destroyAllWindows()
