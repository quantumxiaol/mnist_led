import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pythoncode.network.SimpleNeuralNetwork import SimpleNeuralNetwork
from pythoncode.training.trainer import create_trainer_for_mnist, train_model
from pythoncode.config import Config
import torch.optim as optim

def main():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 创建训练器
    trainer = create_trainer_for_mnist(
        model_class=SimpleNeuralNetwork,
        model_config={'input_size': 784, 'hidden_size': 128, 'output_size': 10},
        batch_size=64,
        learning_rate=0.1,
        optimizer_class=optim.SGD,  # 使用SGD优化器
        data_dir='./data',
        device=device
    )
    
    # 训练模型
    stats = train_model(
        trainer=trainer,
        num_epochs=40,
        save_path='./saved_weights/simple_nn_mnist.pth'
    )
    
    # 保存训练历史
    # trainer.save_training_history('./saved_weights/simple_nn_training_history.json')

if __name__ == '__main__':
    main()