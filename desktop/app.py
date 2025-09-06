import sys
from PySide6.QtWidgets import QApplication
from windows.main_window import QualityControlGUI

def main():
    app = QApplication(sys.argv)
    window = QualityControlGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
