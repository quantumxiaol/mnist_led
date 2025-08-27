# train_lenet.py
from pythoncode.network.LeNet import LeNet5
from pythoncode.training.trainer import create_trainer_for_mnist, train_model
from pythoncode.config import Config
device = Config.DEVICE
if __name__ == '__main__':
    # 创建训练器
    trainer = create_trainer_for_mnist(
        model_class=LeNet5,
        batch_size=64,
        learning_rate=0.001,
        optimizer_class='adam',  # 使用字符串指定优化器
        data_dir='./data',
        device=device
    )
    
    # 训练模型
    stats = train_model(
        trainer=trainer,
        num_epochs=20,
        save_path='./saved_weights/lenet_mnist_new.pth'
    )
    
    # 保存训练历史
    # trainer.save_training_history('./saved_weights/lenet_training_history.json')