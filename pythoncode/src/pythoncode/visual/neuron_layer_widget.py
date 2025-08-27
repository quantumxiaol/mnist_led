from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QScrollArea, 
                             QGridLayout, QHBoxLayout)
from PyQt5.QtCore import Qt
from .led_widget import LEDWidget

class NeuronLayerWidget(QWidget):
    def __init__(self, name, neuron_count, max_display=256):
        super().__init__()
        self.name = name
        self.neuron_count = neuron_count
        self.max_display = max_display
        self.leds = []
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # 层名称
        title = QLabel(f"{self.name} ({self.neuron_count} neurons)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # LED网格
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll_content = QWidget()
        grid_layout = QGridLayout(scroll_content)
        grid_layout.setSpacing(2)
        
        # 计算显示的LED数量
        display_count = min(self.neuron_count, self.max_display)
        cols = min(32, display_count)
        rows = (display_count + cols - 1) // cols
        
        # 创建LED
        for i in range(display_count):
            led = LEDWidget(10)
            self.leds.append(led)
            row = i // cols
            col = i % cols
            grid_layout.addWidget(led, row, col)
            
        # 如果有截断，显示提示
        if self.neuron_count > self.max_display:
            info = QLabel(f"... (显示 {self.max_display}/{self.neuron_count} 个神经元)")
            info.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(info)
            
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        self.setLayout(layout)
        
    def update_activations(self, activations):
        """更新神经元激活状态"""
        import torch
        import numpy as np
        
        # 确保是1D张量
        if len(activations.shape) > 1:
            flat_activations = activations.flatten()
        else:
            flat_activations = activations
            
        # 转换为numpy并标准化
        values = flat_activations.cpu().numpy()
        
        # 标准化到0-1范围
        if len(values) > 0:
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(values)
        else:
            normalized = np.array([])
            
        # 更新LED
        for i, led in enumerate(self.leds):
            if i < len(values):
                is_active = abs(values[i]) > 0.1
                intensity = normalized[i] if i < len(normalized) else 0.0
                led.set_active(is_active, intensity)
            else:
                led.set_active(False, 0.0)