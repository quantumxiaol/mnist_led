import sys
import torch
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout, QScrollArea, 
                             QSpinBox, QFrame, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

from .led_widget import LEDWidget
from .neuron_layer_widget import NeuronLayerWidget

class NeuralVisualizer(QMainWindow):
    def __init__(self, model_class, model_weights_path, model_config=None):
        super().__init__()
        self.model_class = model_class
        self.model_weights_path = model_weights_path
        self.model_config = model_config or {}
        self.current_index = 0
        self.activation_hooks = {}  # 存储hook句柄
        
        # 自动播放定时器
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self.auto_play_step)
        self.auto_play_interval = 1000
        
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
        # 根据配置创建模型实例
        if self.model_config:
            self.model = self.model_class(**self.model_config).to(self.device)
        else:
            self.model = self.model_class().to(self.device)
            
        self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        self.model.eval()
        
        # 为模型添加激活值捕获hook
        self._setup_activation_hooks()
        
    def _setup_activation_hooks(self):
        """为模型的所有可hook层添加激活值捕获"""
        self.model.feature_maps = {}
        
        def create_hook(name):
            def hook_fn(module, input, output):
                self.model.feature_maps[name] = output.clone()
            return hook_fn
        
        # 为所有有参数的层添加hook
        for name, module in self.model.named_modules():
            # 跳过模型本身
            if name == "":
                continue
                
            # 只为有参数的层添加hook（通常是需要可视化的层）
            if hasattr(module, 'weight') and hasattr(module, 'forward'):
                hook = module.register_forward_hook(create_hook(name))
                self.activation_hooks[name] = hook
                
    def init_ui(self):
        self.setWindowTitle('Neural Network 可视化')
        self.setGeometry(50, 50, 1400, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左侧面板
        left_panel = self.create_left_panel()
        # 右侧面板
        right_scroll = self.create_right_panel()
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_scroll, 2)

    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # 输入图像显示
        input_title = QLabel("输入图像 (28×28)")
        input_title.setAlignment(Qt.AlignCenter)
        input_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(input_title)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid black;")
        left_layout.addWidget(self.image_label)
        
        # 输入层神经元显示
        input_neurons_title = QLabel("输入层神经元 (784)")
        input_neurons_title.setAlignment(Qt.AlignCenter)
        input_neurons_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(input_neurons_title)
        
        # 创建28x28的LED网格
        self.input_neurons_widget = QWidget()
        input_neurons_layout = QGridLayout(self.input_neurons_widget)
        input_neurons_layout.setSpacing(1)
        self.input_leds = []
        
        # 创建28x28的LED矩阵
        for i in range(28):
            for j in range(28):
                led = LEDWidget(8)
                self.input_leds.append(led)
                input_neurons_layout.addWidget(led, i, j)
        
        # 包装在滚动区域中
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(350)
        scroll.setWidget(self.input_neurons_widget)
        left_layout.addWidget(scroll)
        
        # 结果显示
        self.result_label = QLabel("点击推理查看结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        left_layout.addWidget(self.result_label)
        
        # 控制按钮
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
        
        # 自动播放控件
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        button_layout.addWidget(separator)
        
        self.auto_play_btn = QPushButton('自动播放')
        self.auto_play_btn.setCheckable(True)
        self.auto_play_btn.clicked.connect(self.toggle_auto_play)
        button_layout.addWidget(self.auto_play_btn)
        
        rate_label = QLabel("间隔 (ms):")
        button_layout.addWidget(rate_label)
        
        self.rate_spinbox = QSpinBox()
        self.rate_spinbox.setRange(100, 10000)
        self.rate_spinbox.setValue(self.auto_play_interval)
        self.rate_spinbox.valueChanged.connect(self.update_auto_play_rate)
        button_layout.addWidget(self.rate_spinbox)
        
        left_layout.addLayout(button_layout)
        left_layout.addStretch()
        
        return left_panel

    def create_right_panel(self):
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        self.layers_layout = QVBoxLayout(right_content)
        right_scroll.setWidget(right_content)
        
        self.layer_widgets = {}
        
        # 根据模型自动创建层显示
        self.create_layer_widgets()
        
        self.layers_layout.addStretch()
        return right_scroll

    def create_layer_widgets(self):
        """根据模型自动创建层显示控件 - 完全自动"""
        self.layer_widgets = {}
        
        # 获取所有hooked的层名称
        hooked_layers = list(self.activation_hooks.keys())
        
        # 按照在模型中的顺序排序
        named_modules = list(self.model.named_modules())
        module_order = {name: i for i, (name, _) in enumerate(named_modules)}
        hooked_layers.sort(key=lambda x: module_order.get(x, len(module_order)))
        
        # 为每个hooked层创建显示控件
        for layer_name in hooked_layers:
            # 获取该层的激活值示例（通过一次前向传播）
            try:
                sample_activation = self._get_sample_activation(layer_name)
                if sample_activation is not None:
                    total_neurons = sample_activation.numel()
                    layer_info = self._get_layer_info(layer_name, sample_activation)
                    
                    widget = NeuronLayerWidget(
                        layer_info['name'],
                        total_neurons,
                        layer_info['max_display']
                    )
                    self.layer_widgets[layer_name] = widget
                    self.layers_layout.addWidget(widget)
            except Exception as e:
                print(f"Warning: Could not create widget for layer {layer_name}: {e}")
                continue

    def _get_sample_activation(self, layer_name):
        """获取指定层的示例激活值"""
        # 使用测试数据集的第一张图片进行一次推理来获取激活值
        try:
            sample_image, _ = self.test_dataset[0]
            sample_input = sample_image.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                self.model(sample_input)
                
            if hasattr(self.model, 'feature_maps') and layer_name in self.model.feature_maps:
                return self.model.feature_maps[layer_name]
        except Exception:
            pass
        return None

    def _get_layer_info(self, layer_name, activation_tensor):
        """根据层名称和激活张量生成显示信息"""
        # 获取模块对象
        module = dict(self.model.named_modules()).get(layer_name, None)
        
        # 生成层名称
        if module:
            module_type = type(module).__name__
            display_name = f"{module_type}: {layer_name}"
        else:
            display_name = f"Layer: {layer_name}"
            
        # 添加形状信息
        shape_str = "×".join(map(str, activation_tensor.shape[1:]))  # 跳过batch维度
        display_name += f" ({shape_str})"
        
        # 计算神经元总数
        total_neurons = activation_tensor.numel()
        
        # 设置最大显示数量
        max_display = min(total_neurons, 256)
        
        return {
            'name': display_name,
            'neurons': total_neurons,
            'max_display': max_display
        }

    def toggle_auto_play(self):
        if self.auto_play_btn.isChecked():
            self.start_auto_play()
        else:
            self.stop_auto_play()

    def start_auto_play(self):
        self.auto_play_btn.setText('停止播放')
        interval = self.rate_spinbox.value()
        self.auto_play_timer.start(interval)
        self.auto_play_step()

    def stop_auto_play(self):
        self.auto_play_timer.stop()
        self.auto_play_btn.setText('自动播放')
        self.auto_play_btn.setChecked(False)

    def update_auto_play_rate(self):
        if self.auto_play_timer.isActive():
            self.auto_play_timer.stop()
            interval = self.rate_spinbox.value()
            self.auto_play_timer.start(interval)

    def auto_play_step(self):
        self.next_image()
        self.run_inference()

    def closeEvent(self, event):
        # 清理hook
        for hook in self.activation_hooks.values():
            hook.remove()
        self.stop_auto_play()
        super().closeEvent(event)
        
    def show_sample(self):
        image, label = self.test_dataset[self.current_index]
        self.current_image = image
        self.current_label = label
        
        # 显示原始图像
        img_np = image.squeeze().numpy()
        img_np = (img_np * 0.3081 + 0.1307)
        img_np = np.clip(img_np, 0, 1)
        img_array = (img_np * 255).astype(np.uint8)
        
        height, width = img_array.shape
        qimage = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.result_label.setText(f"样本索引: {self.current_index}, 真实标签: {label}")
        
    def update_input_neurons(self, image_tensor):
        flat_image = image_tensor.flatten()
        values = flat_image.numpy()
        
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
            
        for i, led in enumerate(self.input_leds):
            if i < len(values):
                intensity = normalized[i]
                led.set_active(True, intensity)
            else:
                led.set_active(False, 0.0)
        
    def run_inference(self):
        if not hasattr(self, 'current_image'):
            return
            
        input_image = self.current_image.unsqueeze(0).to(self.device)
        self.update_input_neurons(self.current_image)
        
        with torch.no_grad():
            output = self.model(input_image)
            
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
        
        self.result_label.setText(f"真实: {self.current_label}, 预测: {predicted} (置信度: {confidence:.2f})")
        
        # 更新各层激活状态
        self.update_layer_activations()
        
    def update_layer_activations(self):
        """更新所有层的激活状态"""
        feature_maps = getattr(self.model, 'feature_maps', {})
        
        for layer_key, widget in self.layer_widgets.items():
            if layer_key in feature_maps:
                activations = feature_maps[layer_key]
                widget.update_activations(activations)
        
    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.test_dataset)
        self.show_sample()
        
    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.test_dataset)
        self.show_sample()

def create_visualizer(model_class, model_weights_path, model_config=None, window_title="Neural Network Visualizer"):
    """创建可视化器的便捷函数"""
    class CustomVisualizer(NeuralVisualizer):
        def init_ui(self):
            super().init_ui()
            self.setWindowTitle(window_title)
    
    app = QApplication(sys.argv)
    window = CustomVisualizer(model_class, model_weights_path, model_config)
    return app, window