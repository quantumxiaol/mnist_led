import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pythoncode.network.SimpleNeuralNetwork import SimpleNeuralNetwork
from pythoncode.config import Config


# ----------------------------
# 1. 设置设备
# ----------------------------
device = Config.DEVICE
print(f"Using device: {device}")

# ----------------------------
# 2. 数据加载 (与SimpleNeuralNetwork匹配)
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平为784维向量，与训练时保持一致
])

test_dataset = MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"Test dataset size: {len(test_dataset)}")

# ----------------------------
# 3. 加载模型权重
# ----------------------------
model = SimpleNeuralNetwork().to(device)
model_path = './saved_weights/simple_nn_mnist.pth'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"权重文件未找到: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 切换到评估模式
print(f"模型权重已从 {model_path} 加载。")

# ----------------------------
# 4. 评估模型
# ----------------------------
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0

print("开始评估...")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"评估完成！")
print(f"Test Loss: {avg_loss:.6f}")
print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")

# ----------------------------
# 5. 可视化几个预测结果
# ----------------------------
def show_sample_predictions(model, test_dataset, num_samples=8):
    """
    显示一些测试样本的预测结果
    """
    model.eval()
    # 随机选择样本索引
    indices = torch.randint(0, len(test_dataset), (num_samples,))
    samples = [test_dataset[i] for i in indices]
    images, labels = zip(*samples)
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 将展平的图像重新reshape为28x28
    images = images.cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        img = images[i].numpy()
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"T:{true_label}, P:{pred_label}", 
                         color='green' if true_label == pred_label else 'red')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 6. 显示详细分类报告
# ----------------------------
def show_classification_report(model, test_loader, num_classes=10):
    """
    显示每个类别的分类准确率
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 计算每个类别的准确率
            correct = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    print("\n各类别准确率:")
    print("-" * 20)
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"数字 {i}: {class_correct[i]}/{class_total[i]} ({accuracy:.1f}%)")
        else:
            print(f"数字 {i}: 0/0 (0.0%)")

# 显示详细分类报告
show_classification_report(model, test_loader)

# 询问是否显示样例
show_sample = input("\n是否显示一些预测样例？(y/n): ").strip().lower()
if show_sample == 'y':
    show_sample_predictions(model, test_dataset)

print("\n=== 评估总结 ===")
print(f"模型类型: SimpleNeuralNetwork (784 → 128 → 10)")
print(f"测试准确率: {accuracy:.2f}%")
print(f"测试损失: {avg_loss:.6f}")