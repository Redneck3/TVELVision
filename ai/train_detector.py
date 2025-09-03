import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading


class TabletDetectorApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Tablet Detector - YOLOv8")
        self.root.geometry("900x700")

        # YOLO model
        self.model = YOLO(model_path)

        # Camera
        self.cap = None
        self.running = False

        # Video / Image label
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start Camera", command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.load_btn = ttk.Button(btn_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=2, padx=10)

        self.quit_btn = ttk.Button(btn_frame, text="Quit", command=self.quit_app)
        self.quit_btn.grid(row=0, column=3, padx=10)
        # FPS label
        self.fps_label = ttk.Label(self.root, text="FPS: 0")
        self.fps_label.pack()

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(2)
            if not self.cap.isOpened():
                print("Камера не найдена")
                return
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_detection(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def quit_app(self):
        self.stop_detection()
        self.root.quit()
        self.root.destroy()

    def update_frame(self):
        prev_time = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLO detection
            results = self.model(frame, imgsz=640, conf=0.1)
            annotated_frame = results[0].plot()

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            self.fps_label.config(text=f"FPS: {int(fps)}")

            # Convert to ImageTk
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        if self.cap:
            self.cap.release()

    def load_image(self):
        """Загрузка фото и проверка YOLO"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            print("Ошибка загрузки изображения")
            return

        # YOLO detection
        results = self.model(img, imgsz=320, conf=0.1)
        annotated_frame = results[0].plot()

        # Convert to ImageTk
        img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        self.fps_label.config(text="Фото загружено")


if __name__ == "__main__":
    root = tk.Tk()
    app = TabletDetectorApp(root, model_path="./runs/detect/train3/weights/best.pt")
    root.mainloop()
