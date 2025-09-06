STYLE_SHEET = """
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
QPushButton:hover { background-color: #bcd8dc; }
QPushButton:pressed { background-color: #8aaeb3; }
QPushButton:disabled { background-color: #d0d8da; color: #7f8c8d; }
QProgressBar {
    border: 1px solid #d0c9c0;
    border-radius: 4px;
    text-align: center;
    background: #f5f0eb;
    height: 20px;
}
QProgressBar::chunk { background-color: #a5d6a7; border-radius: 2px; }
QLCDNumber { background-color: #4a4238; color: #f5f0eb; border-radius: 6px; }
QLabel { color: #4a4238; }
QTableWidget { background-color: #ffffff; gridline-color: #d0c9c0; border-radius: 4px; }
QListWidget { background-color: #ffffff; border: 1px solid #d0c9c0; border-radius: 4px; }
QComboBox, QDoubleSpinBox, QSpinBox { background-color: #ffffff; border: 1px solid #d0c9c0; border-radius: 4px; padding: 5px; color: #4a4238; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #4a4238; }
QCheckBox { color: #4a4238; spacing: 5px; }
QCheckBox::indicator { width: 18px; height: 18px; }
"""
