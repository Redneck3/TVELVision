import sys
import random
import cv2
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, 
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QProgressBar, QLCDNumber, QCheckBox,
    QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QSplitter, QGraphicsView, QGraphicsScene,
    QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox,
    QGraphicsRectItem, QGroupBox, QStackedWidget, QSizePolicy, QSpinBox
)
from PySide6.QtGui import (
    QPixmap, QColor, QPainter, QBrush, QPen, QPainterPath, 
    QFont, QLinearGradient, QImage
)
from PySide6.QtCore import (
    Qt, QTimer, QRectF, QObject, Signal, QPointF, 
    QVariantAnimation, QEasingCurve, QSize
)

class QualityControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система контроля качества таблеток")
        self.setGeometry(100, 100, 1600, 900)
        
        # Цветовая схема
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f0eb;
                color: #4a4238;
            }
            QTabWidget::pane {
                border: 1px solid #d0c9c0;
                background: #f5f0eb;
            }
            QTabBar::tab {
                background: #e5dfd9;
                border: 1px solid #d0c9c0;
                padding: 8px;
                border-radius: 4px;
                color: #4a4238;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #d0c9c0;
                border-bottom-color: #d0c9c0;
            }
            QFrame, QGroupBox {
                background-color: #f9f7f4;
                border: 1px solid #d0c9c0;
                border-radius: 8px;
                padding: 10px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #6b5e4a;
            }
            QPushButton {
                background-color: #a8c7cb;
                border: 1px solid #8aaeb3;
                border-radius: 6px;
                padding: 8px 15px;
                color: #2c3e50;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #bcd8dc;
            }
            QPushButton:pressed {
                background-color: #8aaeb3;
            }
            QPushButton:disabled {
                background-color: #d0d8da;
                color: #7f8c8d;
            }
            QProgressBar {
                border: 1px solid #d0c9c0;
                border-radius: 4px;
                text-align: center;
                background: #f5f0eb;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #a5d6a7;
                border-radius: 2px;
            }
            QLCDNumber {
                background-color: #4a4238;
                color: #f5f0eb;
                border-radius: 6px;
            }
            QLabel {
                color: #4a4238;
            }
            QTableWidget {
                background-color: #ffffff;
                gridline-color: #d0c9c0;
                border-radius: 4px;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #d0c9c0;
                border-radius: 4px;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #ffffff;
                border: 1px solid #d0c9c0;
                border-radius: 4px;
                padding: 5px;
                color: #4a4238;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #4a4238;
            }
            QCheckBox {
                color: #4a4238;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        
        # Папка для сохранения изображений брака
        self.defect_folder = "defect_images"
        if not os.path.exists(self.defect_folder):
            os.makedirs(self.defect_folder)
        
        # Инициализация переменных
        self.current_part = 0
        self.parts_processed = 0
        self.total_parts = 0
        self.defect_count = 0
        self.total_processed = 0
        self.total_defect = 0
        self.is_running = False
        self.auto_next_batch = False
        self.spawn_timer = QTimer()
        self.spawn_timer.timeout.connect(self.spawn_part)
        self.spawn_interval = 800  # Интервал спавна деталей
        
        # Статистика по дефектам поверхности (накапливается)
        self.total_scratches = 0
        self.total_chips = 0
        self.total_cracks = 0
        self.total_other_defects = 0
        
        # История партий
        self.batch_history = []
        self.elapsed_time = 0.0

        ### CAMERA: поля камеры
        self.cap = None
        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self.update_camera_frame)
        self.camera_index = 0  # если нужно, поменяешь на другую камеру

        # Центральный виджет и вкладки
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Вкладки
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Вкладка 1: Контроль качества
        self.quality_tab = QWidget()
        self.tab_widget.addTab(self.quality_tab, "Контроль качества")
        
        # Вкладка 2: История и аналитика
        self.history_tab = QWidget()
        self.tab_widget.addTab(self.history_tab, "История и аналитика")
        
        # Вкладка 3: Настройки
        self.settings_tab = QWidget()
        self.tab_widget.addTab(self.settings_tab, "Настройки")
        
        # Инициализация вкладок
        self.setup_quality_tab()
        self.setup_history_tab()
        self.setup_settings_tab()

        ### CAMERA: автозапуск камеры после создания UI
        QTimer.singleShot(0, self.init_camera)
        
        # Таймер для обновления данных конвейера/статистики
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(30)  # Частое обновление для плавной анимации

    ### CAMERA: инициализация камеры
    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW для Windows ускоряет старт
            if not self.cap.isOpened():
                self.cam_image.setText("❌ Камера недоступна")
                self.cam_image.setAlignment(Qt.AlignCenter)
                return
            # Настройки (по желанию)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1600)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.cam_timer.start(30)  # ~33 кадра/с
        except Exception as e:
            self.cam_image.setText(f"Ошибка запуска камеры: {e}")

    ### CAMERA: обновление кадра
    def update_camera_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # иногда камера «просыпается» пару тактов
            return
        # Переворот/преобразование BGR->RGB
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Подгоним размер под QLabel, сохранив пропорции
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Масштаб к размеру виджета (cover/contain — выберем contain)
        target = self.cam_image.size()
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.cam_image.setPixmap(pix)

    ### CAMERA: остановка и освобождение
    def close_camera(self):
        if self.cam_timer.isActive():
            self.cam_timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

    ### CAMERA: гарантированно освобождаем при закрытии окна
    def closeEvent(self, event):
        self.close_camera()
        super().closeEvent(event)

    def setup_quality_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Верхняя часть: камеры и конвейер
        top_layout = QHBoxLayout()
        
        # Левая колонка: видеокамеры (единая область)
        cameras_group = QGroupBox("Система видеоконтроля")
        cameras_layout = QVBoxLayout(cameras_group)
        
        # Единая область для видеопотока
        cam_frame = QFrame()
        cam_frame.setFrameShape(QFrame.StyledPanel)
        cam_frame.setMinimumSize(400, 300)
        
        # Градиентный фон
        cam_frame.setStyleSheet(f"""
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #e8e0d9, stop:1 #d0c9c0);
            border-radius: 6px;
        """)
        
        cam_layout = QVBoxLayout(cam_frame)
        
        # Изображение камеры
        self.cam_image = QLabel()
        self.cam_image.setAlignment(Qt.AlignCenter)
        self.cam_image.setStyleSheet("""
            background-color: rgba(255, 255, 255, 100); 
            min-height: 280px; 
            border-radius: 4px;
            border: 1px dashed #8aaeb3;
        """)
        self.cam_image.setText("Запуск камеры…")
        
        cam_layout.addWidget(self.cam_image)
        cameras_layout.addWidget(cam_frame)
        top_layout.addWidget(cameras_group, 50)  # 50% ширины

        # Правая колонка: конвейер и управление
        right_layout = QVBoxLayout()
        
        # Конвейер
        conveyor_group = QGroupBox("Конвейерная линия")
        conveyor_layout = QVBoxLayout(conveyor_group)
        
        self.conveyor_view = ConveyorVisualizer()
        conveyor_layout.addWidget(self.conveyor_view)
        right_layout.addWidget(conveyor_group, 70)  # 70% высоты
        
        # Управление
        control_group = QGroupBox("Управление процессом")
        control_layout = QHBoxLayout(control_group)
        
        self.start_btn = QPushButton("▶ СТАРТ")
        self.stop_btn = QPushButton("⏹ СТОП")
        self.save_btn = QPushButton("💾 СОХРАНИТЬ КАДРЫ")
        
        # Стили кнопок
        self.start_btn.setStyleSheet("background-color: #a5d6a7; color: #2c3e50;")
        self.stop_btn.setStyleSheet("background-color: #ef9a9a; color: #2c3e50;")
        self.save_btn.setStyleSheet("background-color: #a8c7cb; color: #2c3e50;")
        
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.save_btn.clicked.connect(self.save_defect_images)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        right_layout.addWidget(control_group, 30)  # 30% высоты
        
        top_layout.addLayout(right_layout, 50)  # 50% ширины
        layout.addLayout(top_layout, 60)  # 60% высоты
        
        # Нижняя часть: статистика
        bottom_layout = QHBoxLayout()
        
        # Статистика партии
        batch_stats_group = QGroupBox("Статистика партии")
        batch_stats_layout = QVBoxLayout(batch_stats_group)
        
        self.batch_label = QLabel(f"ПАРТИЯ: #{self.current_part}")
        self.batch_label.setStyleSheet("""
            font-weight: bold; 
            font-size: 16px; 
            color: #4a4238; 
            background-color: #e5dfd9; 
            padding: 5px; 
            border-radius: 4px;
        """)
        self.batch_label.setAlignment(Qt.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        stats_grid = QHBoxLayout()
        
        left_stats = QVBoxLayout()
        self.defect_percent = QLabel("Брак в партии: 0%")
        self.defect_percent.setStyleSheet("font-size: 14px;")
        self.time_label = QLabel("Время обработки: 0.0 с/деталь")
        self.time_label.setStyleSheet("font-size: 14px;")
        left_stats.addWidget(self.defect_percent)
        left_stats.addWidget(self.time_label)
        
        right_stats = QVBoxLayout()
        defect_label = QLabel("Количество брака:")
        defect_label.setStyleSheet("font-weight: bold;")
        self.defect_lcd = QLCDNumber()
        self.defect_lcd.setDigitCount(3)
        self.defect_lcd.display(0)
        self.defect_lcd.setFixedHeight(60)
        self.defect_lcd.setStyleSheet("background-color: #4a4238; color: #ef9a9a;")
        right_stats.addWidget(defect_label)
        right_stats.addWidget(self.defect_lcd)
        
        stats_grid.addLayout(left_stats)
        stats_grid.addLayout(right_stats)
        
        batch_stats_layout.addWidget(self.batch_label)
        batch_stats_layout.addWidget(self.progress_bar)
        batch_stats_layout.addLayout(stats_grid)
        bottom_layout.addWidget(batch_stats_group, 50)  # 50% ширины
        
        # Общая статистика
        overall_stats_group = QGroupBox("Общая статистика")
        overall_stats_layout = QVBoxLayout(overall_stats_group)
        
        total_layout = QHBoxLayout()
        
        total_left = QVBoxLayout()
        self.total_processed_label = QLabel("Всего обработано: 0")
        self.total_processed_label.setStyleSheet("font-size: 14px;")
        self.total_defect_label = QLabel("Всего брака: 0")
        self.total_defect_label.setStyleSheet("font-size: 14px; color: #d32f2f;")
        total_left.addWidget(self.total_processed_label)
        total_left.addWidget(self.total_defect_label)
        
        total_right = QVBoxLayout()
        self.defect_rate = QLabel("Общий процент брака: 0.0%")
        self.defect_rate.setStyleSheet("font-weight: bold; font-size: 16px; color: #d32f2f;")
        self.parts_per_hour = QLabel("Производительность: 0 дет/час")
        self.parts_per_hour.setStyleSheet("font-size: 14px;")
        total_right.addWidget(self.defect_rate)
        total_right.addWidget(self.parts_per_hour)
        
        total_layout.addLayout(total_left)
        total_layout.addLayout(total_right)
        
        # Показатели дефектов поверхности (накопленные)
        defects_layout = QHBoxLayout()
        
        defects_left = QVBoxLayout()
        defects_title = QLabel("ДЕФЕКТЫ ПОВЕРХНОСТИ")
        defects_title.setStyleSheet("""
            font-weight: bold; 
            text-align: center; 
            background-color: #e5dfd9; 
            padding: 3px; 
            border-radius: 4px;
        """)
        self.scratches_label = QLabel("Царапины: 0")
        self.chips_label = QLabel("Сколы: 0")
        self.cracks_label = QLabel("Трещины: 0")
        self.other_defects_label = QLabel("Другие дефекты: 0")
        defects_left.addWidget(defects_title)
        defects_left.addWidget(self.scratches_label)
        defects_left.addWidget(self.chips_label)
        defects_left.addWidget(self.cracks_label)
        defects_left.addWidget(self.other_defects_label)
        
        defects_layout.addLayout(defects_left)
        
        overall_stats_layout.addLayout(total_layout)
        overall_stats_layout.addLayout(defects_layout)
        bottom_layout.addWidget(overall_stats_group, 50)  # 50% ширины
        
        layout.addLayout(bottom_layout, 40)  # 40% высоты
        
        self.quality_tab.setLayout(layout)

    def setup_history_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Графики
        charts_group = QGroupBox("Аналитика качества")
        charts_layout = QVBoxLayout(charts_group)
        
        # График производительности
        perf_group = QGroupBox("Производительность")
        perf_layout = QVBoxLayout(perf_group)
        self.perf_chart_label = QLabel()
        self.perf_chart_label.setAlignment(Qt.AlignCenter)
        self.perf_chart_label.setStyleSheet("""
            background-color: #ffffff; 
            min-height: 150px; 
            border-radius: 4px;
            border: 1px solid #d0c9c0;
        """)
        self.perf_chart_label.setText("Данные производительности появятся после обработки первой партии")
        perf_layout.addWidget(self.perf_chart_label)
        
        # График качества
        quality_group = QGroupBox("Качество продукции")
        quality_layout = QVBoxLayout(quality_group)
        self.quality_chart_label = QLabel()
        self.quality_chart_label.setAlignment(Qt.AlignCenter)
        self.quality_chart_label.setStyleSheet("""
            background-color: #ffffff; 
            min-height: 150px; 
            border-radius: 4px;
            border: 1px solid #d0c9c0;
        """)
        self.quality_chart_label.setText("Данные качества появятся после обработки первой партии")
        quality_layout.addWidget(self.quality_chart_label)
        
        charts_layout.addWidget(perf_group)
        charts_layout.addWidget(quality_group)
        layout.addWidget(charts_group, 40)  # 40% высоты
        
        # Лог партий
        log_group = QGroupBox("История партий")
        log_layout = QVBoxLayout(log_group)
        
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("font-size: 12px;")
        self.log_list.setSelectionMode(QAbstractItemView.NoSelection)
        
        log_layout.addWidget(self.log_list)
        layout.addWidget(log_group, 60)  # 60% высоты
        
        self.history_tab.setLayout(layout)

    def setup_settings_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Методы оценки
        methods_group = QGroupBox("Методы оценки качества")
        methods_layout = QVBoxLayout(methods_group)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Метод оценки:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["YOLOv8", "U-Net", "Faster R-CNN", "ResNet", "EfficientDet"])
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог детекции:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.1)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        
        methods_layout.addLayout(method_layout)
        methods_layout.addLayout(threshold_layout)
        
        layout.addWidget(methods_group)
        
        # Камеры
        cameras_group = QGroupBox("Настройки камер")
        cameras_layout = QVBoxLayout(cameras_group)
        
        self.cam1_checkbox = QCheckBox("Камера 1 активна")
        self.cam1_checkbox.setChecked(True)
        self.cam2_checkbox = QCheckBox("Камера 2 активна")
        self.cam2_checkbox.setChecked(True)
        self.cam3_checkbox = QCheckBox("Камера 3 активна")
        self.cam3_checkbox.setChecked(True)
        self.auto_save = QCheckBox("Автоматическое сохранение кадров брака")
        self.auto_next_checkbox = QCheckBox("Автоматическая смена партий")
        self.auto_next_checkbox.stateChanged.connect(self.toggle_auto_next)
        
        cameras_layout.addWidget(self.cam1_checkbox)
        cameras_layout.addWidget(self.cam2_checkbox)
        cameras_layout.addWidget(self.cam3_checkbox)
        cameras_layout.addWidget(self.auto_save)
        cameras_layout.addWidget(self.auto_next_checkbox)
        
        layout.addWidget(cameras_group)
        
        # Геометрические настройки
        geometry_group = QGroupBox("Геометрические параметры")
        geom_layout = QVBoxLayout(geometry_group)
        
        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(QLabel("Диаметр, мм:"))
        self.diameter_min = QDoubleSpinBox()
        self.diameter_min.setRange(0.1, 100.0)
        self.diameter_min.setValue(10.0)
        self.diameter_min.setSingleStep(0.1)
        diameter_layout.addWidget(self.diameter_min)
        diameter_layout.addWidget(QLabel("до"))
        self.diameter_max = QDoubleSpinBox()
        self.diameter_max.setRange(0.1, 100.0)
        self.diameter_max.setValue(15.0)
        self.diameter_max.setSingleStep(0.1)
        diameter_layout.addWidget(self.diameter_max)
        
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Длина, мм:"))
        self.length_min = QDoubleSpinBox()
        self.length_min.setRange(0.1, 100.0)
        self.length_min.setValue(20.0)
        self.length_min.setSingleStep(0.1)
        length_layout.addWidget(self.length_min)
        length_layout.addWidget(QLabel("до"))
        self.length_max = QDoubleSpinBox()
        self.length_max.setRange(0.1, 100.0)
        self.length_max.setValue(25.0)
        self.length_max.setSingleStep(0.1)
        length_layout.addWidget(self.length_max)
        
        warp_layout = QHBoxLayout()
        warp_layout.addWidget(QLabel("Макс. искривление, %:"))
        self.warp_spin = QDoubleSpinBox()
        self.warp_spin.setRange(0.1, 10.0)
        self.warp_spin.setValue(1.0)
        self.warp_spin.setSingleStep(0.1)
        warp_layout.addWidget(self.warp_spin)
        warp_layout.addStretch()
        
        geom_layout.addLayout(diameter_layout)
        geom_layout.addLayout(length_layout)
        geom_layout.addLayout(warp_layout)
        
        layout.addWidget(geometry_group)
        
        # Конвейерные настройки
        conveyor_group = QGroupBox("Параметры конвейера")
        conv_layout = QVBoxLayout(conveyor_group)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Скорость конвейера, мм/с:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 100.0)
        self.speed_spin.setValue(40.0)
        self.speed_spin.setSingleStep(1.0)
        speed_layout.addWidget(self.speed_spin)
        
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Расстояние между деталями, мм:"))
        self.part_distance_spin = QDoubleSpinBox()
        self.part_distance_spin.setRange(0.1, 100.0)
        self.part_distance_spin.setValue(20.0)
        self.part_distance_spin.setSingleStep(0.5)
        distance_layout.addWidget(self.part_distance_spin)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Размер партии:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(25)
        batch_layout.addWidget(self.batch_size_spin)
        
        conv_layout.addLayout(speed_layout)
        conv_layout.addLayout(distance_layout)
        conv_layout.addLayout(batch_layout)
        
        layout.addWidget(conveyor_group)
        
        self.settings_tab.setLayout(layout)

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
            self.process_time = 0.0
            self.conveyor_view.reset_positions()
            self.update_defect_labels()
            self.conveyor_view.spawn_new_part()
            self.spawn_timer.start(self.spawn_interval)
            self.start_time = datetime.now()
            if self.auto_next_batch:
                QTimer.singleShot(100, lambda: self.spawn_timer.start(self.spawn_interval))

    def stop_processing(self):
        if self.is_running:
            self.is_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.spawn_timer.stop()
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.elapsed_time = elapsed
            
            # Обновляем общую статистику
            self.total_processed += self.parts_processed
            self.total_defect += self.defect_count
            
            # Добавляем результат партии в лог с подсветкой
            item = QListWidgetItem()
            if self.defect_count > 0:
                item.setText(f"Партия #{self.current_part}: ❌ {self.defect_count} брак из {self.total_parts} | Время: {elapsed:.1f} с")
                item.setForeground(QColor("#d32f2f"))
            else:
                item.setText(f"Партия #{self.current_part}: ✅ Все детали годные | Время: {elapsed:.1f} с")
                item.setForeground(QColor("#388e3c"))
            self.log_list.addItem(item)
            self.log_list.scrollToBottom()
            
            # Сохраняем данные партии в историю
            batch_data = {
                "id": self.current_part,
                "total": self.total_parts,
                "defect": self.defect_count,
                "time": elapsed,
                "timestamp": datetime.now()
            }
            self.batch_history.append(batch_data)
            
            # Обновляем графики
            self.update_charts()
            
            # Сброс конвейера и мягкий автозапуск
            self.conveyor_view.reset_positions()
            QTimer.singleShot(300, self.start_processing)

    def save_defect_images(self):
        files = [f for f in os.listdir(self.defect_folder) if f.endswith(".png")]
        if files:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"Сохранено {len(files)} изображений брака в папку '{self.defect_folder}'")
            msg.exec_()
        else:
            QMessageBox.information(self, "Информация", "Нет изображений брака для сохранения")

    def update_defect_labels(self):
        """Обновление меток с дефектами поверхности"""
        self.scratches_label.setText(f"Царапины: {self.total_scratches}")
        self.chips_label.setText(f"Сколы: {self.total_chips}")
        self.cracks_label.setText(f"Трещины: {self.total_cracks}")
        self.other_defects_label.setText(f"Другие дефекты: {self.total_other_defects}")

    def update_charts(self):
        """Обновление графиков на вкладке истории"""
        if not self.batch_history:
            return
        perf_img = self.create_performance_chart()
        if perf_img:
            self.perf_chart_label.setPixmap(perf_img)
        quality_img = self.create_quality_chart()
        if quality_img:
            self.quality_chart_label.setPixmap(quality_img)

    def create_performance_chart(self):
        """Создание графика производительности"""
        width, height = 600, 150
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Оси
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(30, 20, 30, height - 20)
        painter.drawLine(30, height - 20, width - 10, height - 20)
        
        # Подписи
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(10, 15, "дет/ч")
        painter.drawText(width - 20, height - 5, "Партии")
        
        # Значения
        values = []
        max_value = 1
        for batch in self.batch_history:
            parts_per_hour = (batch["total"] / batch["time"]) * 3600 if batch["time"] > 0 else 0
            values.append(parts_per_hour)
            if parts_per_hour > max_value:
                max_value = parts_per_hour
        max_value = max_value * 1.2 if max_value > 0 else 100
        
        if values:
            x_step = (width - 50) / len(values)
            painter.setPen(QPen(QColor("#4a4238"), 2))
            points = []
            for i, value in enumerate(values):
                x = 40 + i * x_step
                y = height - 20 - (value / max_value) * (height - 40)
                points.append(QPointF(x, y))
                painter.setBrush(QBrush(QColor("#a8c7cb")))
                painter.drawEllipse(QPointF(x, y), 4, 4)
                painter.drawText(x - 10, height - 5, f"{i+1}")
            painter.drawPolyline(points)
            for i, value in enumerate(values):
                x = 40 + i * x_step
                painter.drawText(x - 15, height - 25 - (values[i] / max_value) * (height - 40), f"{int(values[i])}")
        painter.end()
        return pixmap

    def create_quality_chart(self):
        """Создание графика качества"""
        width, height = 600, 150
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(30, 20, 30, height - 20)
        painter.drawLine(30, height - 20, width - 10, height - 20)
        
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(10, 15, "% брака")
        painter.drawText(width - 20, height - 5, "Партии")
        
        values = [(b["defect"] / b["total"]) * 100 if b["total"] > 0 else 0 for b in self.batch_history]
        if values:
            x_step = (width - 50) / len(values)
            painter.setPen(QPen(QColor("#d32f2f"), 2))
            points = []
            for i, value in enumerate(values):
                x = 40 + i * x_step
                y = height - 20 - (value / 100) * (height - 40)
                points.append(QPointF(x, y))
                painter.setBrush(QBrush(QColor("#ef9a9a")))
                painter.drawEllipse(QPointF(x, y), 4, 4)
                painter.drawText(x - 10, height - 5, f"{i+1}")
            painter.drawPolyline(points)
            for i, value in enumerate(values):
                x = 40 + i * x_step
                painter.drawText(x - 10, height - 25 - (values[i] / 100) * (height - 40), f"{values[i]:.1f}%")
        painter.end()
        return pixmap

    def update_data(self):
        if self.is_running:
            self.conveyor_view.update_positions(self.speed_spin.value())
            processed = self.conveyor_view.process_parts(
                self,
                self.threshold_spin.value(),
                self.diameter_min.value(),
                self.diameter_max.value(),
                self.length_min.value(),
                self.length_max.value(),
                self.warp_spin.value()
            )
            for part in processed:
                self.parts_processed += 1
                if part["defect"]:
                    self.defect_count += 1
                    if self.auto_save.isChecked():
                        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_defect.png"
                        with open(os.path.join(self.defect_folder, filename), 'wb') as f:
                            f.write(b'fake_image_data')
                progress = (self.parts_processed / self.total_parts) * 100
                self.progress_bar.setValue(int(progress))
                elapsed = (datetime.now() - self.start_time).total_seconds()
                time_per_part = elapsed / self.parts_processed if self.parts_processed > 0 else 0
                defect_percent = (self.defect_count / self.parts_processed) * 100 if self.parts_processed > 0 else 0
                self.defect_percent.setText(f"Брак в партии: {defect_percent:.1f}%")
                self.defect_lcd.display(self.defect_count)
                self.time_label.setText(f"Время обработки: {time_per_part:.1f} с/деталь")
                self.total_scratches += part["scratches"]
                self.total_chips += part["chips"]
                self.total_cracks += part["cracks"]
                self.total_other_defects += part["other_defects"]
                self.update_defect_labels()
                self.total_processed_label.setText(f"Всего обработано: {self.total_processed + self.parts_processed}")
                self.total_defect_label.setText(f"Всего брака: {self.total_defect + self.defect_count}")
                total_processed = self.total_processed + self.parts_processed
                total_defect = self.total_defect + self.defect_count
                defect_rate = (total_defect / total_processed) * 100 if total_processed > 0 else 0
                self.defect_rate.setText(f"Общий процент брака: {defect_rate:.1f}%")
                parts_per_hour = self.parts_processed / elapsed * 3600 if elapsed > 0 else 0
                self.parts_per_hour.setText(f"Производительность: {int(parts_per_hour)} дет/час")
            if self.parts_processed >= self.total_parts and len(self.conveyor_view.positions) == 0:
                self.stop_processing()

    def spawn_part(self):
        if self.is_running and self.parts_processed + len(self.conveyor_view.positions) <= self.total_parts:
            self.conveyor_view.spawn_new_part()

class ConveyorVisualizer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFixedHeight(500)
        self.setMinimumWidth(800)
        self.setRenderHint(QPainter.Antialiasing)
        self.positions = []
        self.max_positions = 10
        
        # Счётчики для корзин (общие, сохраняются между партиями)
        self.ok_count = 0
        self.defect_count = 0
        
        # Рисуем конвейер (слева направо)
        self.setBackgroundBrush(QBrush(QColor("#e5dfd9")))
        
        # Конвейер до зоны контроля
        self.conveyor_path1 = QPainterPath()
        self.conveyor_path1.moveTo(0, 260)
        self.conveyor_path1.lineTo(350, 260)
        self.scene.addPath(self.conveyor_path1, QPen(QColor("#4a4238"), 4))
        
        # Конвейер после зоны контроля
        self.conveyor_path2 = QPainterPath()
        self.conveyor_path2.moveTo(450, 260)
        self.conveyor_path2.lineTo(800, 260)
        self.scene.addPath(self.conveyor_path2, QPen(QColor("#4a4238"), 4))
        
        # Конвейер до зоны контроля (нижний)
        self.conveyor_path3 = QPainterPath()
        self.conveyor_path3.moveTo(0, 310)
        self.conveyor_path3.lineTo(350, 310)
        self.scene.addPath(self.conveyor_path3, QPen(QColor("#4a4238"), 4))
        
        # Конвейер после зоны контроля (нижний)
        self.conveyor_path4 = QPainterPath()
        self.conveyor_path4.moveTo(450, 310)
        self.conveyor_path4.lineTo(800, 310)
        self.scene.addPath(self.conveyor_path4, QPen(QColor("#4a4238"), 4))

        # Зона контроля (в центре)
        self.control_zone = QGraphicsRectItem(350, 110, 100, 350)
        self.control_zone.setBrush(QBrush(QColor(100, 181, 246, 100)))
        self.control_zone.setPen(QPen(QColor(33, 150, 243), 2, Qt.DashLine))
        self.scene.addItem(self.control_zone)
        
        # Текст "Зона контроля" в две строки
        text1 = self.scene.addText("ЗОНА")
        text1.setPos(375, 235)
        text1.setDefaultTextColor(QColor(33, 150, 243))
        font = text1.font()
        font.setBold(True)
        font.setPointSize(10)
        text1.setFont(font)
        
        text2 = self.scene.addText("КОНТРОЛЯ")
        text2.setPos(360, 255)
        text2.setDefaultTextColor(QColor(33, 150, 243))
        text2.setFont(font)
        
        # Корзина для годных деталей (справа сверху)
        self.ok_bin = QGraphicsRectItem(550, 50, 120, 100)
        self.ok_bin.setBrush(QBrush(QColor("#a5d6a7")))
        self.ok_bin.setPen(QPen(QColor("#388e3c"), 2))
        self.scene.addItem(self.ok_bin)
        
        # Счётчик годных деталей внутри корзины
        self.ok_counter = self.scene.addText("0")
        self.ok_counter.setPos(595, 80)
        self.ok_counter.setDefaultTextColor(QColor("#388e3c"))
        font = self.ok_counter.font()
        font.setPointSize(20)
        font.setBold(True)
        self.ok_counter.setFont(font)
        self.ok_counter.setPlainText(str(self.ok_count))
        
        # Корзина для бракованных деталей (справа снизу)
        self.defect_bin = QGraphicsRectItem(550, 410, 120, 100)
        self.defect_bin.setBrush(QBrush(QColor("#ef9a9a")))
        self.defect_bin.setPen(QPen(QColor("#d32f2f"), 2))
        self.scene.addItem(self.defect_bin)
        
        # Счётчик бракованных деталей внутри корзины
        self.defect_counter = self.scene.addText("0")
        self.defect_counter.setPos(595, 435)
        self.defect_counter.setDefaultTextColor(QColor("#d32f2f"))
        font = self.defect_counter.font()
        font.setPointSize(20)
        font.setBold(True)
        self.defect_counter.setFont(font)
        self.defect_counter.setPlainText(str(self.defect_count))
        
    def reset_positions(self):
        """Сброс только позиций на конвейере, не счетчиков"""
        for pos in self.positions[:]:
            if pos["item"]:
                self.scene.removeItem(pos["item"])
            if pos in self.positions:
                self.positions.remove(pos)
                
    def spawn_new_part(self):
        if len(self.positions) < self.max_positions:
            # Стартовая позиция - слева
            start_x = -50
            new_pos = {
                "x": start_x,
                "y": 138,
                "status": "processing",
                "defect": False,
                "item": None
            }
            rect = QRectF(new_pos["x"], new_pos["y"], 40, 20)
            path = QPainterPath()
            path.addRoundedRect(rect, 5, 5)
            color = QColor("#90caf9")
            item = self.scene.addPath(path, QPen(QColor("#4a4238"), 1), QBrush(color))
            item.setOpacity(1)
            new_pos["item"] = item
            self.positions.append(new_pos)
        
    def update_positions(self, speed):
        for pos in self.positions:
            pos["x"] += speed / 8
            if pos["item"]:
                pos["item"].setPos(pos["x"], pos["y"])
            
    def process_parts(self, party, threshold, diam_min, diam_max, len_min, len_max, warp_max):
        processed = []
        positions_copy = self.positions.copy()
        for pos in positions_copy:
            if 350 <= pos["x"] <= 450 and pos["status"] == "processing":
                diameter = random.uniform(diam_min - 0.5, diam_max + 0.5)
                length = random.uniform(len_min - 1, len_max + 1)
                warp = random.uniform(0.0, max(warp_max, 0.001) * 1.2)
                
                scratches = 1 if random.random() < 0.1 and party.defect_count < 2 else 0
                chips = 1 if random.random() < 0.05 and party.defect_count < 2 else 0
                cracks = 1 if random.random() < 0.03 and party.defect_count < 2 else 0
                other_defects = 1 if random.random() < 0.02 and party.defect_count < 2 else 0
                
                defect = False
                defect_prob = 0.05 + (abs(diameter - (diam_min+diam_max)/2)/(max(diam_max-diam_min,0.001)) * 0.1)
                defect_prob += (abs(length - (len_min+len_max)/2)/(max(len_max-len_min,0.001)) * 0.05)
                defect_prob += (min(warp / max(warp_max,0.001), 1.0) * 0.05)
                if scratches or chips or cracks:
                    defect_prob += 0.2
                defect_prob = min(defect_prob, 0.9)
                if random.random() < defect_prob and party.defect_count < 2:
                    defect = True
                
                pos["status"] = "processed"
                pos["defect"] = defect
                if defect:
                    self.defect_count += 1
                    target_x, target_y = 610, 330
                    color = QColor("#ef9a9a")
                else:
                    self.ok_count += 1
                    target_x, target_y = 610, 80
                    color = QColor("#a5d6a7")
                
                self.defect_counter.setPlainText(str(self.defect_count))
                self.ok_counter.setPlainText(str(self.ok_count))
                pos["item"].setBrush(QBrush(color))
                
                anim = QVariantAnimation()
                anim.setDuration(800)
                anim.setStartValue(QPointF(pos["x"], pos["y"]))
                anim.setEndValue(QPointF(target_x, target_y))
                anim.setEasingCurve(QEasingCurve.OutCubic)
                def update_position(value, p=pos):
                    if p["item"]:
                        p["item"].setPos(value)
                anim.valueChanged.connect(update_position)
                anim.start()
                QTimer.singleShot(850, lambda p=pos: self.remove_part(p))
                
                processed.append({
                    "defect": defect,
                    "diameter": diameter,
                    "length": length,
                    "warp": warp,
                    "scratches": scratches,
                    "chips": chips,
                    "cracks": cracks,
                    "other_defects": other_defects
                })
        return processed

    def remove_part(self, part):
        """Удаление детали из сцены и списка"""
        if part in self.positions:
            if part["item"]:
                self.scene.removeItem(part["item"])
            self.positions.remove(part)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QualityControlGUI()
    window.show()
    sys.exit(app.exec())
