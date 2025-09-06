import os
from datetime import datetime

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, 
    QVBoxLayout, QLabel, QPushButton, 
    QComboBox, QProgressBar, QLCDNumber, QCheckBox,
    QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QSplitter, QGraphicsView, QGraphicsScene,
    QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox,
    QGraphicsRectItem, QGroupBox, QStackedWidget, QSizePolicy, QSpinBox
)



from ui.styles import STYLE_SHEET
from core.config import (
    DEFECT_FOLDER, SPAWN_INTERVAL_MS, UI_TIMER_MS, CAM_TIMER_MS, DEFAULT_CAMERA_INDEX
)
from core.camera import CameraManager
from detection.yolo_detection import YoloDetector
from widgets.conveyor import ConveyorVisualizer
from charts.simple_charts import performance_chart, quality_chart


class QualityControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система контроля качества таблеток")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet(STYLE_SHEET)

        # ФС 
        if not os.path.exists(DEFECT_FOLDER):
            os.makedirs(DEFECT_FOLDER)

        # Состояния
        self.current_part = 0
        self.parts_processed = 0
        self.total_parts = 0
        self.defect_count = 0
        self.total_processed = 0
        self.total_defect = 0
        self.is_running = False
        self.auto_next_batch = False
        self.elapsed_time = 0.0

        # Дефекты (накопительные)
        self.total_scratches = 0
        self.total_chips = 0
        self.total_cracks = 0
        self.total_other_defects = 0

        # История
        self.batch_history = []

        # Камера
        self.cam_timer = QTimer(); self.cam_timer.timeout.connect(self.update_camera_frame)
        self.camera_index = DEFAULT_CAMERA_INDEX
        self.camera = None  # CameraManager

        # Таймеры
        self.spawn_timer = QTimer(); self.spawn_timer.timeout.connect(self.spawn_part)
        self.timer = QTimer(); self.timer.timeout.connect(self.update_data); self.timer.start(UI_TIMER_MS)

        # Вкладки
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tab_widget = QTabWidget(); self.main_layout.addWidget(self.tab_widget)
        self.quality_tab = QWidget(); self.history_tab = QWidget(); self.settings_tab = QWidget()
        self.tab_widget.addTab(self.quality_tab, "Контроль качества")
        self.tab_widget.addTab(self.history_tab, "История и аналитика")
        self.tab_widget.addTab(self.settings_tab, "Настройки")

        # Построение вкладок
        self._setup_quality_tab()
        self._setup_history_tab()
        self._setup_settings_tab()

        # Модель
        self.detector = YoloDetector()

    # ---------- Камера ----------
    def init_camera(self):
        self.camera = CameraManager(self.camera_index, status_label=self.cam_image)
        if self.camera.open():
            self.cam_timer.start(CAM_TIMER_MS)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def update_camera_frame(self):
        if not self.camera:
            return
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return

        results = self.detector.predict(frame, imgsz=640, conf=0.085)
        annotated = results[0].plot()  # BGR

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        target = self.cam_image.size()
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_image.setPixmap(pix)

    def close_camera(self):
        if self.cam_timer.isActive():
            self.cam_timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def closeEvent(self, event):
        self.close_camera()
        super().closeEvent(event)

    # ---------- Вкладки ----------
    def _setup_quality_tab(self):
        layout = QVBoxLayout(); layout.setSpacing(15); layout.setContentsMargins(15, 15, 15, 15)
        top = QHBoxLayout()

        # Левая колонка — камера
        cameras_group = QGroupBox("Система видеоконтроля"); cameras_layout = QVBoxLayout(cameras_group)
        cam_frame = QFrame(); cam_frame.setFrameShape(QFrame.StyledPanel); cam_frame.setMinimumSize(200, 100)
        cam_frame.setStyleSheet("""
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e8e0d9, stop:1 #d0c9c0);
            border-radius: 6px;
        """)
        cam_layout = QVBoxLayout(cam_frame)
        self.cam_image = QLabel(); self.cam_image.setAlignment(Qt.AlignCenter)
        self.cam_image.setStyleSheet("""
            background-color: rgba(255, 255, 255, 100);
            min-height: 280px; border-radius: 4px; border: 1px dashed #8aaeb3;
        """)
        self.cam_image.setText("Ожидания запуска программы, нажмите СТАРТ")
        cam_layout.addWidget(self.cam_image)
        cameras_layout.addWidget(cam_frame)
        top.addWidget(cameras_group, 50)

        # Правая колонка — конвейер и управление
        right = QVBoxLayout()
        conveyor_group = QGroupBox("Конвейерная линия"); conveyor_layout = QVBoxLayout(conveyor_group)
        self.conveyor_view = ConveyorVisualizer(); 
        conveyor_layout.addWidget(self.conveyor_view)
        right.addWidget(conveyor_group, 70)

        control_group = QGroupBox("Управление процессом"); control_layout = QHBoxLayout(control_group)
        self.start_btn = QPushButton("▶ СТАРТ"); self.stop_btn = QPushButton("⏹ СТОП"); self.save_btn = QPushButton("💾 СОХРАНИТЬ КАДРЫ")
        self.start_btn.setStyleSheet("background-color: #a5d6a7; color: #2c3e50;")
        self.stop_btn.setStyleSheet("background-color: #ef9a9a; color: #2c3e50;")
        self.save_btn.setStyleSheet("background-color: #a8c7cb; color: #2c3e50;")
        self.start_btn.clicked.connect(self.init_camera)
        self.stop_btn.clicked.connect(self.close_camera)
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.save_btn.clicked.connect(self.save_defect_images)
        self.stop_btn.setEnabled(False)
        for b in (self.start_btn, self.stop_btn, self.save_btn):
            control_layout.addWidget(b)
        right.addWidget(control_group, 30)

        top.addLayout(right, 50)
        layout.addLayout(top, 60)

        # Нижняя часть — статистика
        bottom = QHBoxLayout()
        # Статистика партии
        batch_stats = QGroupBox("Статистика партии"); batch_layout = QVBoxLayout(batch_stats)
        self.batch_label = QLabel(f"ПАРТИЯ: #{getattr(self, 'current_part', 0)}")
        self.batch_label.setStyleSheet("""
            font-weight: bold; font-size: 16px; color: #4a4238; background-color: #e5dfd9; padding: 5px; border-radius: 4px;
        """); self.batch_label.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        stats_grid = QHBoxLayout()
        left_stats = QVBoxLayout(); self.defect_percent = QLabel("Брак в партии: 0%"); self.defect_percent.setStyleSheet("font-size: 14px;")
        self.time_label = QLabel("Время обработки: 0.0 с/деталь"); self.time_label.setStyleSheet("font-size: 14px;")
        left_stats.addWidget(self.defect_percent); left_stats.addWidget(self.time_label)
        right_stats = QVBoxLayout(); defect_label = QLabel("Количество брака:"); defect_label.setStyleSheet("font-weight: bold;")
        self.defect_lcd = QLCDNumber(); self.defect_lcd.setDigitCount(3); self.defect_lcd.display(0); self.defect_lcd.setFixedHeight(60)
        self.defect_lcd.setStyleSheet("background-color: #4a4238; color: #ef9a9a;")
        right_stats.addWidget(defect_label); right_stats.addWidget(self.defect_lcd)
        stats_grid.addLayout(left_stats); stats_grid.addLayout(right_stats)
        batch_layout.addWidget(self.batch_label); batch_layout.addWidget(self.progress_bar); batch_layout.addLayout(stats_grid)
        bottom.addWidget(batch_stats, 50)

        # Общая статистика
        overall = QGroupBox("Общая статистика"); overall_layout = QVBoxLayout(overall)
        total_layout = QHBoxLayout()
        total_left = QVBoxLayout(); self.total_processed_label = QLabel("Всего обработано: 0"); self.total_processed_label.setStyleSheet("font-size: 14px;")
        self.total_defect_label = QLabel("Всего брака: 0"); self.total_defect_label.setStyleSheet("font-size: 14px; color: #d32f2f;")
        total_left.addWidget(self.total_processed_label); total_left.addWidget(self.total_defect_label)
        total_right = QVBoxLayout(); self.defect_rate = QLabel("Общий процент брака: 0.0%"); self.defect_rate.setStyleSheet("font-weight: bold; font-size: 16px; color: #d32f2f;")
        self.parts_per_hour = QLabel("Производительность: 0 дет/час"); self.parts_per_hour.setStyleSheet("font-size: 14px;")
        total_right.addWidget(self.defect_rate); total_right.addWidget(self.parts_per_hour)
        total_layout.addLayout(total_left); total_layout.addLayout(total_right)
        defects_left = QVBoxLayout(); defects_title = QLabel("ДЕФЕКТЫ ПОВЕРХНОСТИ")
        defects_title.setStyleSheet("font-weight: bold; text-align: center; background-color: #e5dfd9; padding: 3px; border-radius: 4px;")
        self.scratches_label = QLabel("Царапины: 0"); self.chips_label = QLabel("Сколы: 0"); self.cracks_label = QLabel("Трещины: 0"); self.other_defects_label = QLabel("Другие дефекты: 0")
        for w in (defects_title, self.scratches_label, self.chips_label, self.cracks_label, self.other_defects_label):
            defects_left.addWidget(w)
        overall_layout.addLayout(total_layout); overall_layout.addLayout(defects_left)
        bottom.addWidget(overall, 50)

        layout.addLayout(bottom, 40)
        self.quality_tab.setLayout(layout)

    def _setup_history_tab(self):
        layout = QVBoxLayout(); layout.setSpacing(15); layout.setContentsMargins(15, 15, 15, 15)
        charts_group = QGroupBox("Аналитика качества"); charts_layout = QVBoxLayout(charts_group)
        perf_group = QGroupBox("Производительность"); perf_layout = QVBoxLayout(perf_group)
        self.perf_chart_label = QLabel(); self.perf_chart_label.setAlignment(Qt.AlignCenter)
        self.perf_chart_label.setStyleSheet("background-color: #ffffff; min-height: 150px; border-radius: 4px; border: 1px solid #d0c9c0;")
        self.perf_chart_label.setText("Данные производительности появятся после обработки первой партии")
        perf_layout.addWidget(self.perf_chart_label)

        quality_group = QGroupBox("Качество продукции"); quality_layout = QVBoxLayout(quality_group)
        self.quality_chart_label = QLabel(); self.quality_chart_label.setAlignment(Qt.AlignCenter)
        self.quality_chart_label.setStyleSheet("background-color: #ffffff; min-height: 150px; border-radius: 4px; border: 1px solid #d0c9c0;")
        self.quality_chart_label.setText("Данные качества появятся после обработки первой партии")
        quality_layout.addWidget(self.quality_chart_label)

        charts_layout.addWidget(perf_group); charts_layout.addWidget(quality_group)
        layout.addWidget(charts_group, 40)

        log_group = QGroupBox("История партий"); log_layout = QVBoxLayout(log_group)
        self.log_list = QListWidget(); self.log_list.setStyleSheet("font-size: 12px;"); self.log_list.setSelectionMode(QAbstractItemView.NoSelection)
        log_layout.addWidget(self.log_list)
        layout.addWidget(log_group, 60)
        self.history_tab.setLayout(layout)

    def _setup_settings_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        cameras_group = QGroupBox("Настройки камер"); cameras_layout = QVBoxLayout(cameras_group)
        self.cam1_checkbox = QCheckBox("Камера 1 активна"); self.cam1_checkbox.setChecked(True)
        self.cam2_checkbox = QCheckBox("Камера 2 активна"); self.cam2_checkbox.setChecked(True)
        self.cam3_checkbox = QCheckBox("Камера 3 активна"); self.cam3_checkbox.setChecked(True)
        self.auto_save = QCheckBox("Автоматическое сохранение кадров брака")
        self.auto_next_checkbox = QCheckBox("Автоматическая смена партий"); self.auto_next_checkbox.stateChanged.connect(self.toggle_auto_next)
        for w in (self.cam1_checkbox, self.cam2_checkbox, self.cam3_checkbox, self.auto_save, self.auto_next_checkbox):
            cameras_layout.addWidget(w)
        layout.addWidget(cameras_group)
        
        

        geometry_group = QGroupBox("Геометрические параметры"); geom_layout = QVBoxLayout(geometry_group)
        from PySide6.QtWidgets import QHBoxLayout  # локально, чтобы не засорять импорт
        diameter_layout = QHBoxLayout(); diameter_layout.addWidget(QLabel("Диаметр, мм:"))
        self.diameter_min = QDoubleSpinBox(); self.diameter_min.setRange(0.1, 100.0); self.diameter_min.setValue(10.0); self.diameter_min.setSingleStep(0.1)
        diameter_layout.addWidget(self.diameter_min); diameter_layout.addWidget(QLabel("до"))
        self.diameter_max = QDoubleSpinBox(); self.diameter_max.setRange(0.1, 100.0); self.diameter_max.setValue(15.0); self.diameter_max.setSingleStep(0.1)
        diameter_layout.addWidget(self.diameter_max)
        length_layout = QHBoxLayout(); length_layout.addWidget(QLabel("Длина, мм:"))
        self.length_min = QDoubleSpinBox(); self.length_min.setRange(0.1, 100.0); self.length_min.setValue(20.0); self.length_min.setSingleStep(0.1)
        length_layout.addWidget(self.length_min); length_layout.addWidget(QLabel("до"))
        self.length_max = QDoubleSpinBox(); self.length_max.setRange(0.1, 100.0); self.length_max.setValue(25.0); self.length_max.setSingleStep(0.1)
        length_layout.addWidget(self.length_max)
        warp_layout = QHBoxLayout(); warp_layout.addWidget(QLabel("Макс. искривление, %:"))
        self.warp_spin = QDoubleSpinBox(); self.warp_spin.setRange(0.1, 10.0); self.warp_spin.setValue(1.0); self.warp_spin.setSingleStep(0.1)
        warp_layout.addWidget(self.warp_spin); warp_layout.addStretch()
        for lay in (diameter_layout, length_layout, warp_layout):
            geom_layout.addLayout(lay)
        layout.addWidget(geometry_group)

        conveyor_group = QGroupBox("Параметры конвейера"); conv_layout = QVBoxLayout(conveyor_group)
        speed_layout = QHBoxLayout(); speed_layout.addWidget(QLabel("Скорость конвейера, мм/с:"))
        self.speed_spin = QDoubleSpinBox(); self.speed_spin.setRange(0.1, 100.0); self.speed_spin.setValue(40.0); self.speed_spin.setSingleStep(1.0)
        speed_layout.addWidget(self.speed_spin)
        distance_layout = QHBoxLayout(); distance_layout.addWidget(QLabel("Расстояние между деталями, мм:"))
        self.part_distance_spin = QDoubleSpinBox(); self.part_distance_spin.setRange(0.1, 100.0); self.part_distance_spin.setValue(20.0); self.part_distance_spin.setSingleStep(0.5)
        distance_layout.addWidget(self.part_distance_spin)
        batch_layout = QHBoxLayout(); batch_layout.addWidget(QLabel("Размер партии:"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 100); self.batch_size_spin.setValue(25)
        batch_layout.addWidget(self.batch_size_spin)
        for lay in (speed_layout, distance_layout, batch_layout):
            conv_layout.addLayout(lay)
        layout.addWidget(conveyor_group)

        self.settings_tab.setLayout(layout)

    # ---------- Логика процесса ----------
    def toggle_auto_next(self, state):
        self.auto_next_batch = state == Qt.Checked

    def start_processing(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.total_parts = self.batch_size_spin.value()
            self.parts_processed = 0
            self.defect_count = 0
            self.current_part += 1
            self.batch_label.setText(f"ПАРТИЯ: #{self.current_part}")
            self.progress_bar.setValue(0)
            self.defect_lcd.display(0)
            self.conveyor_view.reset_positions()
            self.update_defect_labels()
            self.conveyor_view.spawn_new_part()
            self.spawn_timer.start(SPAWN_INTERVAL_MS)
            self.start_time = datetime.now()
            if self.auto_next_batch:
                QTimer.singleShot(200, lambda: self.spawn_timer.start(SPAWN_INTERVAL_MS))

    def stop_processing(self):
        if self.is_running:
            self.is_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.spawn_timer.stop()
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.elapsed_time = elapsed
            self.total_processed += self.parts_processed
            self.total_defect += self.defect_count

            item = QListWidgetItem()
            if self.defect_count > 0:
                item.setText(f"Партия #{self.current_part}: ❌ {self.defect_count} брак из {self.total_parts} | Время: {elapsed:.1f} с")
                item.setForeground(QColor("#d32f2f"))
            else:
                item.setText(f"Партия #{self.current_part}: ✅ Все детали годные | Время: {elapsed:.1f} с")
                item.setForeground(QColor("#388e3c"))
            self.log_list.addItem(item); self.log_list.scrollToBottom()

            self.batch_history.append({
                "id": self.current_part,
                "total": self.total_parts,
                "defect": self.defect_count,
                "time": elapsed,
                "timestamp": datetime.now(),
            })
            self.update_charts()
            self.conveyor_view.reset_positions()
            QTimer.singleShot(300, self.start_processing)

    def save_defect_images(self):
        files = [f for f in os.listdir(DEFECT_FOLDER) if f.endswith(".png")]
        from PySide6.QtWidgets import QMessageBox
        if files:
            msg = QMessageBox(); msg.setIcon(QMessageBox.Information)
            msg.setText(f"Сохранено {len(files)} изображений брака в папку '{DEFECT_FOLDER}'"); msg.exec_()
        else:
            QMessageBox.information(self, "Информация", "Нет изображений брака для сохранения")

    def update_defect_labels(self):
        self.scratches_label.setText(f"Царапины: {self.total_scratches}")
        self.chips_label.setText(f"Сколы: {self.total_chips}")
        self.cracks_label.setText(f"Трещины: {self.total_cracks}")
        self.other_defects_label.setText(f"Другие дефекты: {self.total_other_defects}")

    def update_charts(self):
        if not self.batch_history:
            return
        self.perf_chart_label.setPixmap(performance_chart(self.batch_history))
        self.quality_chart_label.setPixmap(quality_chart(self.batch_history))

    def update_data(self):
        if self.is_running:
            self.conveyor_view.update_positions(self.speed_spin.value())
            processed = self.conveyor_view.process_parts(
                self,
                self.diameter_min.value(), self.diameter_max.value(),
                self.length_min.value(), self.length_max.value(),
                self.warp_spin.value(),
            )
            for part in processed:
                self.parts_processed += 1
                if part["defect"]:
                    self.defect_count += 1
                    if self.auto_save.isChecked():
                        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_defect.png"
                        with open(os.path.join(DEFECT_FOLDER, filename), 'wb') as f:
                            f.write(b'fake_image_data')
                progress = int((self.parts_processed / self.total_parts) * 100)
                self.progress_bar.setValue(progress)
                elapsed = (datetime.now() - self.start_time).total_seconds()
                tpp = elapsed / self.parts_processed if self.parts_processed > 0 else 0
                dpercent = (self.defect_count / self.parts_processed) * 100 if self.parts_processed > 0 else 0
                self.defect_percent.setText(f"Брак в партии: {dpercent:.1f}%")
                self.defect_lcd.display(self.defect_count)
                self.time_label.setText(f"Время обработки: {tpp:.1f} с/деталь")
                self.total_scratches += part["scratches"]; self.total_chips += part["chips"]; self.total_cracks += part["cracks"]; self.total_other_defects += part["other_defects"]
                self.update_defect_labels()
                self.total_processed_label.setText(f"Всего обработано: {self.total_processed + self.parts_processed}")
                self.total_defect_label.setText(f"Всего брака: {self.total_defect + self.defect_count}")
                tp = self.total_processed + self.parts_processed
                td = self.total_defect + self.defect_count
                rate = (td / tp) * 100 if tp > 0 else 0
                self.defect_rate.setText(f"Общий процент брака: {rate:.1f}%")
                pph = self.parts_processed / elapsed * 3600 if elapsed > 0 else 0
                self.parts_per_hour.setText(f"Производительность: {int(pph)} дет/час")
            if self.parts_processed >= self.total_parts and len(self.conveyor_view.positions) == 0:
                self.stop_processing()

    def spawn_part(self):
        if self.is_running and self.parts_processed + len(self.conveyor_view.positions) <= self.total_parts:
            self.conveyor_view.spawn_new_part()
