from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt

class LEDWidget(QWidget):
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.setFixedSize(size, size)
        self.active = False
        self.intensity = 0.0
        
    def set_active(self, active, intensity=1.0):
        self.active = active
        self.intensity = max(0.0, min(1.0, abs(intensity)))
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.active and self.intensity > 0.1:
            # 根据激活强度设置颜色（从黄到红）
            red = int(255 * self.intensity)
            green = int(255 * (1 - self.intensity))
            color = QColor(red, green, 0)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.black, 1))
        else:
            # 未激活时显示灰色
            painter.setBrush(QBrush(QColor(50, 50, 50)))
            painter.setPen(QPen(Qt.gray, 1))
            
        painter.drawEllipse(1, 1, self.size-2, self.size-2)