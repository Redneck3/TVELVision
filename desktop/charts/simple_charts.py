from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush


def performance_chart(batch_history):
    width, height = 600, 150
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.white)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(Qt.black, 2))
    painter.drawLine(30, 20, 30, height - 20)
    painter.drawLine(30, height - 20, width - 10, height - 20)

    font = painter.font(); font.setPointSize(8); painter.setFont(font)
    painter.drawText(10, 15, "дет/ч"); painter.drawText(width - 20, height - 5, "Партии")

    values = []
    max_value = 1
    for batch in batch_history:
        t = batch.get("time", 0)
        total = batch.get("total", 0)
        v = (total / t) * 3600 if t > 0 else 0
        values.append(v)
        max_value = max(max_value, v)
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


def quality_chart(batch_history):
    width, height = 600, 150
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.white)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(Qt.black, 2))
    painter.drawLine(30, 20, 30, height - 20)
    painter.drawLine(30, height - 20, width - 10, height - 20)

    font = painter.font(); font.setPointSize(8); painter.setFont(font)
    painter.drawText(10, 15, "% брака"); painter.drawText(width - 20, height - 5, "Партии")

    values = [(b.get("defect", 0) / b.get("total", 1)) * 100 for b in batch_history]
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
