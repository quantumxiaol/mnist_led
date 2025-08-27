# evaluate_simple_nn.py
from pythoncode.network.SimpleNeuralNetwork import SimpleNeuralNetworkG
from pythoncode.training.evaluator import create_evaluator_for_mnist, evaluate_model
from pythoncode.config import Config

device = Config.DEVICE

if __name__ == '__main__':
    # 创建评估器
    evaluator = create_evaluator_for_mnist(
        model_class=SimpleNeuralNetworkG,
        model_weights_path='./saved_weights/simple_nn_mnist.pth',
        model_config={'input_size': 784, 'hidden_size': 128, 'output_size': 10},
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
        model_info="SimpleNeuralNetwork (784 → 128 → 10)"
    )