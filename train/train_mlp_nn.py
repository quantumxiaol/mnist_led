import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pythoncode.network.MLPNetwork import MLPNetwork
from pythoncode.training.trainer import create_trainer_for_mnist, train_model
from pythoncode.config import Config
import torch.optim as optim

def main():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 创建训练器 - 使用更深的网络结构
    trainer = create_trainer_for_mnist(
        model_class=MLPNetwork,
        model_config={'input_size': 784, 'hidden_sizes': [64, 64, 32], 'output_size': 10},
        batch_size=64,
        learning_rate=0.01,
        optimizer_class=optim.Adam,  # Adam通常对深层网络更好
        data_dir='./data',
        device=device
    )
    
    # 训练模型
    stats = train_model(
        trainer=trainer,
        num_epochs=50,
        save_path='./saved_weights/mlp_nn_deeper_mnist.pth'
    )
    
    # 保存训练历史
    # trainer.save_training_history('./saved_weights/mlp_nn_deeper_training_history.json')

if __name__ == '__main__':
    main()