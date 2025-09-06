import random
from PySide6.QtCore import Qt, QRectF, QVariantAnimation, QEasingCurve, QPointF, QTimer
from PySide6.QtGui import QColor, QPen, QBrush, QPainterPath, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem

class ConveyorVisualizer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFixedHeight(400)
        self.setMinimumWidth(150)
        self.setRenderHint(QPainter.Antialiasing)
        self.positions = []
        self.max_positions = 10
        self.ok_count = 0
        self.defect_count = 0
        self.setBackgroundBrush(QBrush(QColor("#e5dfd9")))

        # Линии конвейера
        for y in (260, 310):
            path_l = QPainterPath(); path_l.moveTo(0, y); path_l.lineTo(350, y)
            self.scene.addPath(path_l, QPen(QColor("#4a4238"), 4))
            path_r = QPainterPath(); path_r.moveTo(450, y); path_r.lineTo(800, y)
            self.scene.addPath(path_r, QPen(QColor("#4a4238"), 4))

        # Зона контроля
        self.control_zone = QGraphicsRectItem(350, 90, 100, 350)
        self.control_zone.setBrush(QBrush(QColor(100, 181, 246, 100)))
        self.control_zone.setPen(QPen(QColor(33, 150, 243), 2, Qt.DashLine))
        self.scene.addItem(self.control_zone)
        text1 = self.scene.addText("ЗОНА"); text1.setPos(375, 235); text1.setDefaultTextColor(QColor(33, 150, 243))
        f = text1.font(); f.setBold(True); f.setPointSize(10); text1.setFont(f)
        text2 = self.scene.addText("КОНТРОЛЯ"); text2.setPos(360, 255); text2.setDefaultTextColor(QColor(33, 150, 243)); text2.setFont(f)

        # Корзины
        self.ok_bin = QGraphicsRectItem(550, 50, 120, 100)
        self.ok_bin.setBrush(QBrush(QColor("#a5d6a7")))
        self.ok_bin.setPen(QPen(QColor("#388e3c"), 2))
        self.scene.addItem(self.ok_bin)
        self.ok_counter = self.scene.addText("0"); self.ok_counter.setPos(595, 80); self.ok_counter.setDefaultTextColor(QColor("#388e3c"))
        ff = self.ok_counter.font(); ff.setPointSize(20); ff.setBold(True); self.ok_counter.setFont(ff)

        self.defect_bin = QGraphicsRectItem(550, 350, 120, 100)
        self.defect_bin.setBrush(QBrush(QColor("#ef9a9a")))
        self.defect_bin.setPen(QPen(QColor("#d32f2f"), 2))
        self.scene.addItem(self.defect_bin)
        self.defect_counter = self.scene.addText("0"); self.defect_counter.setPos(595, 375); self.defect_counter.setDefaultTextColor(QColor("#d32f2f"))
        f2 = self.defect_counter.font(); f2.setPointSize(20); f2.setBold(True); self.defect_counter.setFont(f2)

    def reset_positions(self):
        for pos in self.positions[:]:
            if pos.get("item"):
                self.scene.removeItem(pos["item"])
            if pos in self.positions:
                self.positions.remove(pos)

    def spawn_new_part(self):
        if len(self.positions) < self.max_positions:
            start_x = -10
            new_pos = {"x": start_x, "y": 138, "status": "processing", "defect": False, "item": None}
            rect = QRectF(new_pos["x"], new_pos["y"], 40, 20)
            path = QPainterPath(); path.addRoundedRect(rect, 5, 5)
            color = QColor("#90caf9")
            item = self.scene.addPath(path, QPen(QColor("#4a4238"), 1), QBrush(color))
            item.setOpacity(1)
            new_pos["item"] = item
            self.positions.append(new_pos)

    def update_positions(self, speed):
        for pos in self.positions:
            pos["x"] += speed / 8
            if pos.get("item"):
                pos["item"].setPos(pos["x"], pos["y"])

    def process_parts(self, party, threshold, diam_min, diam_max, len_min, len_max, warp_max):
        processed = []
        for pos in self.positions.copy():
            if 350 <= pos["x"] <= 450 and pos["status"] == "processing":
                # Псевдо-измерения
                diameter = random.uniform(diam_min - 0.5, diam_max + 0.5)
                length = random.uniform(len_min - 1, len_max + 1)
                warp = random.uniform(0.0, max(warp_max, 0.001) * 1.2)
                scratches = 1 if random.random() < 0.1 and party.defect_count < 2 else 0
                chips = 1 if random.random() < 0.05 and party.defect_count < 2 else 0
                cracks = 1 if random.random() < 0.03 and party.defect_count < 2 else 0
                other_defects = 1 if random.random() < 0.02 and party.defect_count < 2 else 0

                defect_prob = 0.05
                gap_d = abs(diameter - (diam_min + diam_max) / 2) / max((diam_max - diam_min), 0.001)
                gap_l = abs(length - (len_min + len_max) / 2) / max((len_max - len_min), 0.001)
                defect_prob += gap_d * 0.1 + gap_l * 0.05 + min(warp / max(warp_max, 0.001), 1.0) * 0.05
                if scratches or chips or cracks:
                    defect_prob += 0.2
                defect_prob = min(defect_prob, 0.9)
                defect = random.random() < defect_prob and party.defect_count < 2

                pos["status"] = "processed"; pos["defect"] = defect
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
                if pos.get("item"):
                    pos["item"].setBrush(QBrush(color))

                anim = QVariantAnimation(); anim.setDuration(800)
                anim.setStartValue(QPointF(pos["x"], pos["y"]))
                anim.setEndValue(QPointF(target_x, target_y))
                anim.setEasingCurve(QEasingCurve.OutCubic)
                def _update(value, p=pos):
                    if p.get("item"):
                        p["item"].setPos(value)
                anim.valueChanged.connect(_update)
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
                    "other_defects": other_defects,
                })
        return processed

    def remove_part(self, part):
        if part in self.positions:
            if part.get("item"):
                self.scene.removeItem(part["item"])
            self.positions.remove(part)
