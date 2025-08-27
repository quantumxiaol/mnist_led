import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Optional, Tuple, Union
import json

class ModelTrainer:
    def __init__(self, 
                 model_class,
                 model_config: Optional[Dict[str, Any]] = None,
                 dataset_class=None,
                 dataset_config: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 criterion_class=nn.CrossEntropyLoss,
                 criterion_config: Optional[Dict[str, Any]] = None,
                 optimizer_class: Union[str, type] = optim.Adam,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None):
        """
        通用模型训练器
        
        Args:
            model_class: 模型类
            model_config: 模型配置参数
            dataset_class: 数据集类
            dataset_config: 数据集配置参数
            transform: 数据变换
            criterion_class: 损失函数类
            criterion_config: 损失函数配置参数
            optimizer_class: 优化器类（可以是字符串或类）
            optimizer_config: 优化器配置参数
            device: 计算设备
        """
        self.model_class = model_class
        self.model_config = model_config or {}
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config or {}
        self.transform = transform
        self.criterion_class = criterion_class
        self.criterion_config = criterion_config or {}
        self.optimizer_class = self._resolve_optimizer_class(optimizer_class)
        self.optimizer_config = optimizer_config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练状态
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.training_history = []
        
    def _resolve_optimizer_class(self, optimizer_class):
        """解析优化器类"""
        if isinstance(optimizer_class, str):
            optimizer_map = {
                'adam': optim.Adam,
                'sgd': optim.SGD,
                'adamw': optim.AdamW,
                'rmsprop': optim.RMSprop
            }
            return optimizer_map.get(optimizer_class.lower(), optim.Adam)
        return optimizer_class
        
    def setup_data(self, 
                   train_dataset=None, 
                   test_dataset=None,
                   train_loader_config: Optional[Dict[str, Any]] = None,
                   test_loader_config: Optional[Dict[str, Any]] = None):
        """
        设置数据加载器
        
        Args:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            train_loader_config: 训练数据加载器配置
            test_loader_config: 测试数据加载器配置
        """
        train_loader_config = train_loader_config or {'batch_size': 64, 'shuffle': True}
        test_loader_config = test_loader_config or {'batch_size': 64, 'shuffle': False}
        
        # 如果提供了数据集，直接使用
        if train_dataset is not None and test_dataset is not None:
            self.train_loader = DataLoader(train_dataset, **train_loader_config)
            self.test_loader = DataLoader(test_dataset, **test_loader_config)
        # 否则根据配置创建数据集
        elif self.dataset_class is not None:
            # 处理MNIST数据集的特殊情况
            if hasattr(self.dataset_class, '__name__') and 'MNIST' in self.dataset_class.__name__:
                train_dataset = self.dataset_class(transform=self.transform, train=True, **self.dataset_config)
                test_dataset = self.dataset_class(transform=self.transform, train=False, **self.dataset_config)
            else:
                train_dataset = self.dataset_class(transform=self.transform, **self.dataset_config)
                test_dataset = self.dataset_class(transform=self.transform, **self.dataset_config)
            self.train_loader = DataLoader(train_dataset, **train_loader_config)
            self.test_loader = DataLoader(test_dataset, **test_loader_config)
        else:
            raise ValueError("必须提供数据集或数据集类")
            
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        
    def setup_model(self):
        """初始化模型、损失函数和优化器"""
        # 创建模型
        self.model = self.model_class(**self.model_config).to(self.device)
        
        # 创建损失函数
        self.criterion = self.criterion_class(**self.criterion_config)
        
        # 创建优化器
        self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_config)
        
        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_test_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        return {
            'loss': avg_test_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
    def train(self, 
              num_epochs: int = 10,
              save_path: Optional[str] = None,
              save_best_only: bool = True,
              print_every_epoch: bool = True) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_path: 模型保存路径
            save_best_only: 是否只保存最佳模型
            print_every_epoch: 是否每轮都打印信息
            
        Returns:
            训练统计信息
        """
        if self.model is None:
            self.setup_model()
            
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("必须先调用setup_data设置数据加载器")
            
        print("Starting training...")
        start_time = time.time()
        best_accuracy = 0.0
        best_model_state = None
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_stats = self.train_epoch()
            epoch_duration = time.time() - epoch_start_time
            
            # 评估
            eval_stats = self.evaluate()
            
            # 记录历史
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_stats['loss'],
                'train_accuracy': train_stats['accuracy'],
                'test_loss': eval_stats['loss'],
                'test_accuracy': eval_stats['accuracy'],
                'epoch_duration': epoch_duration
            }
            self.training_history.append(epoch_info)
            
            # 保存最佳模型
            if eval_stats['accuracy'] > best_accuracy:
                best_accuracy = eval_stats['accuracy']
                if save_best_only:
                    best_model_state = self.model.state_dict()
            
            # 打印信息
            if print_every_epoch:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_stats["loss"]:.6f}, '
                      f'Train Acc: {train_stats["accuracy"]:.2f}%, '
                      f'Test Loss: {eval_stats["loss"]:.6f}, '
                      f'Test Acc: {eval_stats["accuracy"]:.2f}%')
        
        total_training_time = time.time() - start_time
        print('Training complete.')
        
        # 最终评估
        final_eval = self.evaluate()
        
        # 保存模型
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if save_best_only and best_model_state is not None:
                torch.save(best_model_state, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            print(f'Model weights saved to {save_path}')
        
        # 统计信息
        training_stats = {
            'total_epochs': num_epochs,
            'total_training_time': total_training_time,
            'average_time_per_epoch': total_training_time / num_epochs,
            'final_test_accuracy': final_eval['accuracy'],
            'best_test_accuracy': best_accuracy,
            'final_test_loss': final_eval['loss'],
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_history': self.training_history
        }
        
        self._print_summary(training_stats)
        return training_stats
        
    def _print_summary(self, stats: Dict[str, Any]):
        """打印训练总结"""
        print("\n=== Training Summary ===")
        print(f"Total Epochs: {stats['total_epochs']}")
        print(f"Total Training Time: {stats['total_training_time']:.2f}s")
        print(f"Average Time per Epoch: {stats['average_time_per_epoch']:.2f}s")
        print(f"Final Test Accuracy: {stats['final_test_accuracy']:.2f}%")
        print(f"Best Test Accuracy: {stats['best_test_accuracy']:.2f}%")
        print(f"Model Parameters: {stats['model_parameters']:,}")
        
    def save_training_history(self, filepath: str):
        """保存训练历史到JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f'Training history saved to {filepath}')

# 便捷函数
def create_trainer_for_mnist(model_class,
                           model_config: Optional[Dict[str, Any]] = None,
                           batch_size: int = 64,
                           learning_rate: float = 0.001,
                           optimizer_class: Union[str, type] = optim.Adam,
                           data_dir: str = './data',
                           device: Optional[torch.device] = None) -> ModelTrainer:
    """
    为MNIST数据集创建训练器的便捷函数
    """
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    
    # 默认变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 默认优化器配置
    optimizer_config = {'lr': learning_rate}
    
    trainer = ModelTrainer(
        model_class=model_class,
        model_config=model_config,
        dataset_class=MNIST,
        dataset_config={'root': data_dir, 'download': True},
        transform=transform,
        optimizer_class=optimizer_class,
        optimizer_config=optimizer_config,
        device=device
    )
    
    return trainer

def train_model(trainer: ModelTrainer,
                num_epochs: int = 10,
                batch_size: int = 64,
                save_path: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
    """
    训练模型的便捷函数
    """
    # 设置数据加载器
    trainer.setup_data(
        train_loader_config={'batch_size': batch_size, 'shuffle': True},
        test_loader_config={'batch_size': batch_size, 'shuffle': False}
    )
    
    # 训练模型
    stats = trainer.train(
        num_epochs=num_epochs,
        save_path=save_path,
        **kwargs
    )
    
    return stats