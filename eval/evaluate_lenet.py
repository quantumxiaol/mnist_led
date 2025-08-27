import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pythoncode.network.LeNet import LeNet5
from pythoncode.training.evaluator import create_evaluator_for_mnist, evaluate_model
from pythoncode.config import Config

def main():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 创建评估器
    evaluator = create_evaluator_for_mnist(
        model_class=LeNet5,
        model_weights_path='./saved_weights/lenet_mnist.pth',
        batch_size=64,
        data_dir='./data',
        device=device
    )
    
    # 评估模型
    eval_stats = evaluate_model(
        evaluator=evaluator,
        batch_size=64,
        show_samples=True,      # 显示样本预测
        num_samples=8,          # 显示8个样本
        show_report=True        # 显示详细分类报告
    )
    
    # 打印总结
    evaluator.print_summary(
        eval_stats,
        model_info="LeNet-5"
    )

if __name__ == '__main__':
    main()