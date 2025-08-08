import sys
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from LeNet import LeNet5G
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout, QGridLayout, QScrollArea, QSpinBox, QFrame)

from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QRect, QTimer

# 自定义LED控件
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

# 神经元层显示控件
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

# 主可视化窗口
class NeuronVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_index = 0
        # 新增：自动播放定时器
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self.auto_play_step)
        self.auto_play_interval = 1000 # 默认1000毫秒
        
        self.init_model()
        self.init_ui()
        self.show_sample()
        
    def init_model(self):
        # 初始化模型和数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_dataset = MNIST(root='./data', train=False, download=False, transform=transform)
        
        self.device = torch.device("cpu")
        self.model = LeNet5G().to(self.device)
        self.model.load_state_dict(torch.load('./saved_weights/lenet_mnist.pth', map_location=self.device))
        self.model.eval()
        
    def init_ui(self):
        self.setWindowTitle('LeNet-5 神经元激活可视化')
        self.setGeometry(50, 50, 1400, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        input_title = QLabel("输入图像 (28×28)")
        input_title.setAlignment(Qt.AlignCenter)
        input_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(input_title)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid black;")
        left_layout.addWidget(self.image_label)
        
        input_neurons_title = QLabel("输入层神经元 (784)")
        input_neurons_title.setAlignment(Qt.AlignCenter)
        input_neurons_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(input_neurons_title)
        
        self.input_neurons_widget = QWidget()
        input_neurons_layout = QGridLayout(self.input_neurons_widget)
        input_neurons_layout.setSpacing(1)
        self.input_leds = []
        
        for i in range(28):
            for j in range(28):
                led = LEDWidget(6)
                self.input_leds.append(led)
                input_neurons_layout.addWidget(led, i, j)
                
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        scroll.setWidget(self.input_neurons_widget)
        left_layout.addWidget(scroll)
        
        self.result_label = QLabel("点击推理查看结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        left_layout.addWidget(self.result_label)
        
        # --- 修改控制按钮区域 ---
        button_layout = QHBoxLayout()

        self.prev_btn = QPushButton('上一张')
        self.prev_btn.clicked.connect(self.prev_image)
        button_layout.addWidget(self.prev_btn)
        
        self.inference_btn = QPushButton('推理')
        self.inference_btn.clicked.connect(self.run_inference)
        button_layout.addWidget(self.inference_btn)
        
        self.next_btn = QPushButton('下一张')
        self.next_btn.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_btn)

        # --- 新增自动播放控件 ---
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        button_layout.addWidget(separator)

        self.auto_play_btn = QPushButton('自动播放')
        self.auto_play_btn.setCheckable(True) # 使按钮有按下/弹起状态
        self.auto_play_btn.clicked.connect(self.toggle_auto_play)
        button_layout.addWidget(self.auto_play_btn)

        # 添加速率设置 (毫秒)
        rate_label = QLabel("间隔 (ms):")
        button_layout.addWidget(rate_label)

        self.rate_spinbox = QSpinBox()
        self.rate_spinbox.setRange(100, 10000) # 100ms 到 10s
        self.rate_spinbox.setValue(self.auto_play_interval)
        self.rate_spinbox.valueChanged.connect(self.update_auto_play_rate)
        button_layout.addWidget(self.rate_spinbox)

        left_layout.addLayout(button_layout)
        left_layout.addStretch()
        
        # 右侧：神经元层显示
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        self.layers_layout = QVBoxLayout(right_content)
        right_scroll.setWidget(right_content)
        
        self.layer_widgets = {}
        self.layer_widgets['conv1'] = NeuronLayerWidget('Conv1 特征图 (6×24×24)', 6*24*24, 256)
        self.layers_layout.addWidget(self.layer_widgets['conv1'])
        self.layer_widgets['pool1'] = NeuronLayerWidget('Pool1 特征图 (6×12×12)', 6*12*12, 256)
        self.layers_layout.addWidget(self.layer_widgets['pool1'])
        self.layer_widgets['conv2'] = NeuronLayerWidget('Conv2 特征图 (16×8×8)', 16*8*8, 256)
        self.layers_layout.addWidget(self.layer_widgets['conv2'])
        self.layer_widgets['pool2'] = NeuronLayerWidget('Pool2 特征图 (16×4×4)', 16*4*4, 256)
        self.layers_layout.addWidget(self.layer_widgets['pool2'])
        self.layer_widgets['fc1'] = NeuronLayerWidget('FC1 全连接 (120)', 120, 120)
        self.layers_layout.addWidget(self.layer_widgets['fc1'])
        self.layer_widgets['fc2'] = NeuronLayerWidget('FC2 全连接 (84)', 84, 84)
        self.layers_layout.addWidget(self.layer_widgets['fc2'])
        self.layer_widgets['output'] = NeuronLayerWidget('Output 输出 (10)', 10, 10)
        self.layers_layout.addWidget(self.layer_widgets['output'])
        
        self.layers_layout.addStretch()
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_scroll, 2)

    # --- 新增自动播放相关方法 ---
    
    def toggle_auto_play(self):
        """切换自动播放状态"""
        if self.auto_play_btn.isChecked():
            self.start_auto_play()
        else:
            self.stop_auto_play()

    def start_auto_play(self):
        """开始自动播放"""
        self.auto_play_btn.setText('停止播放')
        interval = self.rate_spinbox.value()
        self.auto_play_timer.start(interval)
        # 立即执行一次
        self.auto_play_step()

    def stop_auto_play(self):
        """停止自动播放"""
        self.auto_play_timer.stop()
        self.auto_play_btn.setText('自动播放')
        self.auto_play_btn.setChecked(False)

    def update_auto_play_rate(self):
        """更新自动播放速率"""
        if self.auto_play_timer.isActive():
            # 如果正在播放，重新启动定时器以应用新速率
            self.auto_play_timer.stop()
            interval = self.rate_spinbox.value()
            self.auto_play_timer.start(interval)

    def auto_play_step(self):
        """自动播放的一步：切换到下一张图片并执行推理"""
        self.next_image() # 切换到下一张
        self.run_inference() # 执行推理

    # 确保在窗口关闭时停止定时器
    def closeEvent(self, event):
        self.stop_auto_play()
        super().closeEvent(event)
        
    def show_sample(self):
        """显示当前样本图像"""
        image, label = self.test_dataset[self.current_index]
        self.current_image = image
        self.current_label = label
        
        # 显示原始图像
        img_np = image.squeeze().numpy()
        # 反归一化
        img_np = (img_np * 0.3081 + 0.1307)
        img_np = np.clip(img_np, 0, 1)
        img_array = (img_np * 255).astype(np.uint8)
        
        # 转换为QPixmap显示
        height, width = img_array.shape
        qimage = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 更新结果标签
        self.result_label.setText(f"样本索引: {self.current_index}, 真实标签: {label}")
        
    def update_input_neurons(self, image_tensor):
        """更新输入层神经元显示"""
        # 展平图像
        flat_image = image_tensor.flatten()
        values = flat_image.numpy()
        
        # 标准化到0-1范围
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
            
        # 更新LED
        for i, led in enumerate(self.input_leds):
            if i < len(values):
                intensity = normalized[i]
                # 输入层所有像素都视为"激活"，只是强度不同
                led.set_active(True, intensity)
            else:
                led.set_active(False, 0.0)
        
    def run_inference(self):
        """执行推理并更新所有层的可视化"""
        if not hasattr(self, 'current_image'):
            return
            
        input_image = self.current_image.unsqueeze(0).to(self.device)
        
        # 更新输入层显示
        self.update_input_neurons(self.current_image)
        
        # 执行推理
        with torch.no_grad():
            output = self.model(input_image)
            
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
        
        # 更新结果
        self.result_label.setText(f"真实: {self.current_label}, 预测: {predicted} (置信度: {confidence:.2f})")
        
        # 更新各层激活状态
        # Conv1层
        conv1_activations = self.model.feature_maps['conv1']
        self.layer_widgets['conv1'].update_activations(conv1_activations)
        
        # Pool1层
        pool1_activations = self.model.feature_maps['pool1']
        self.layer_widgets['pool1'].update_activations(pool1_activations)
        
        # Conv2层
        conv2_activations = self.model.feature_maps['conv2']
        self.layer_widgets['conv2'].update_activations(conv2_activations)
        
        # Pool2层
        pool2_activations = self.model.feature_maps['pool2']
        self.layer_widgets['pool2'].update_activations(pool2_activations)
        
        # FC1层
        fc1_activations = self.model.feature_maps['fc1']
        self.layer_widgets['fc1'].update_activations(fc1_activations)
        
        # FC2层
        fc2_activations = self.model.feature_maps['fc2']
        self.layer_widgets['fc2'].update_activations(fc2_activations)
        
        # 输出层
        output_activations = self.model.feature_maps['output']
        self.layer_widgets['output'].update_activations(output_activations)
        
    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.test_dataset)
        self.show_sample()
        
    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.test_dataset)
        self.show_sample()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeuronVisualizer()
    window.show()
    sys.exit(app.exec_())