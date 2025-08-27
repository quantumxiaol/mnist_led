import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np

class ModelEvaluator:
    def __init__(self, 
                 model_class,
                 model_weights_path: str,
                 model_config: Optional[Dict[str, Any]] = None,
                 dataset_class=None,
                 dataset_config: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 device: Optional[torch.device] = None):
        """
        通用模型评估器
        
        Args:
            model_class: 模型类
            model_weights_path: 模型权重路径
            model_config: 模型配置参数
            dataset_class: 数据集类
            dataset_config: 数据集配置参数
            transform: 数据变换
            device: 计算设备
        """
        self.model_class = model_class
        self.model_weights_path = model_weights_path
        self.model_config = model_config or {}
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config or {}
        self.transform = transform
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 评估组件
        self.model = None
        self.test_loader = None
        self.criterion = nn.CrossEntropyLoss()
        
    def setup_model(self):
        """初始化模型并加载权重"""
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(f"权重文件未找到: {self.model_weights_path}")
            
        # 创建模型
        self.model = self.model_class(**self.model_config).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        self.model.eval()  # 切换到评估模式
        
        print(f"模型权重已从 {self.model_weights_path} 加载。")
        print(f"使用设备: {self.device}")
        
    def setup_data(self, 
                   test_dataset=None,
                   test_loader_config: Optional[Dict[str, Any]] = None):
        """
        设置测试数据加载器
        
        Args:
            test_dataset: 测试数据集
            test_loader_config: 测试数据加载器配置
        """
        test_loader_config = test_loader_config or {'batch_size': 64, 'shuffle': False}
        
        # 如果提供了数据集，直接使用
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, **test_loader_config)
        # 否则根据配置创建数据集
        elif self.dataset_class is not None:
            # 处理MNIST数据集的特殊情况
            if hasattr(self.dataset_class, '__name__') and 'MNIST' in self.dataset_class.__name__:
                test_dataset = self.dataset_class(transform=self.transform, train=False, **self.dataset_config)
            else:
                test_dataset = self.dataset_class(transform=self.transform, **self.dataset_config)
            self.test_loader = DataLoader(test_dataset, **test_loader_config)
        else:
            raise ValueError("必须提供测试数据集或数据集类")
            
        print(f"测试数据集大小: {len(test_dataset)}")
        
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            评估统计信息
        """
        if self.model is None:
            self.setup_model()
        if self.test_loader is None:
            raise ValueError("必须先调用setup_data设置数据加载器")
            
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        print("开始评估...")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        print(f"评估完成！")
        print(f"Test Loss: {avg_loss:.6f}")
        print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
    def show_sample_predictions(self, num_samples: int = 8, figsize: Tuple[int, int] = (12, 3)):
        """
        显示一些测试样本的预测结果
        
        Args:
            num_samples: 显示样本数量
            figsize: 图形大小
        """
        if self.model is None:
            self.setup_model()
            
        # 获取测试数据集
        test_dataset = self.test_loader.dataset
        
        # 随机选择样本索引
        indices = torch.randint(0, len(test_dataset), (num_samples,))
        samples = [test_dataset[i] for i in indices]
        
        # 处理不同格式的数据
        if len(samples[0]) == 2:  # (image, label) 格式
            images, labels = zip(*samples)
            images = torch.stack(images).to(self.device)
            labels = torch.tensor(labels)
        else:
            raise ValueError("不支持的数据格式")
            
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            
        # 处理图像显示
        images_cpu = images.cpu()
        # 如果图像是展平的，尝试reshape
        if len(images_cpu.shape) == 2:
            images_cpu = images_cpu.view(-1, int(np.sqrt(images_cpu.shape[1])), int(np.sqrt(images_cpu.shape[1])))
        elif len(images_cpu.shape) == 4 and images_cpu.shape[1] == 1:
            images_cpu = images_cpu.squeeze(1)
            
        fig, axes = plt.subplots(1, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            img = images_cpu[i].numpy()
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"T:{true_label}, P:{pred_label}", 
                             color='green' if true_label == pred_label else 'red')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def show_classification_report(self, num_classes: int = 10) -> Dict[int, Dict[str, float]]:
        """
        显示每个类别的分类准确率
        
        Args:
            num_classes: 类别数量
            
        Returns:
            各类别详细统计信息
        """
        if self.model is None:
            self.setup_model()
        if self.test_loader is None:
            raise ValueError("必须先调用setup_data设置数据加载器")
            
        self.model.eval()
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                # 计算每个类别的准确率
                correct = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += correct[i].item() if correct.dim() > 0 else correct.item()
                    class_total[label] += 1
                    
        print("\n各类别准确率:")
        print("-" * 20)
        class_stats = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print(f"类别 {i}: {class_correct[i]}/{class_total[i]} ({accuracy:.1f}%)")
                class_stats[i] = {
                    'correct': class_correct[i],
                    'total': class_total[i],
                    'accuracy': accuracy
                }
            else:
                print(f"类别 {i}: 0/0 (0.0%)")
                class_stats[i] = {
                    'correct': 0,
                    'total': 0,
                    'accuracy': 0.0
                }
                
        return class_stats
        
    def print_summary(self, eval_stats: Dict[str, float], model_info: str = ""):
        """打印评估总结"""
        print("\n=== 评估总结 ===")
        if model_info:
            print(f"模型信息: {model_info}")
        print(f"测试准确率: {eval_stats['accuracy']:.2f}%")
        print(f"测试损失: {eval_stats['loss']:.6f}")
        print(f"正确预测: {eval_stats['correct']}/{eval_stats['total']}")

# 便捷函数
def create_evaluator_for_mnist(model_class,
                             model_weights_path: str,
                             model_config: Optional[Dict[str, Any]] = None,
                             batch_size: int = 64,
                             data_dir: str = './data',
                             device: Optional[torch.device] = None) -> ModelEvaluator:
    """
    为MNIST数据集创建评估器的便捷函数
    """
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    
    # 默认变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    evaluator = ModelEvaluator(
        model_class=model_class,
        model_weights_path=model_weights_path,
        model_config=model_config,
        dataset_class=MNIST,
        dataset_config={'root': data_dir, 'download': False},
        transform=transform,
        device=device
    )
    
    return evaluator

def evaluate_model(evaluator: ModelEvaluator,
                   batch_size: int = 64,
                   show_samples: bool = False,
                   num_samples: int = 8,
                   show_report: bool = True) -> Dict[str, Any]:
    """
    评估模型的便捷函数
    """
    # 设置数据加载器
    evaluator.setup_data(
        test_loader_config={'batch_size': batch_size, 'shuffle': False}
    )
    
    # 评估模型
    eval_stats = evaluator.evaluate()
    
    # 显示分类报告
    if show_report:
        evaluator.show_classification_report()
        
    # 显示样本预测
    if show_samples:
        evaluator.show_sample_predictions(num_samples=num_samples)
        
    return eval_stats