import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pythoncode.src.pythoncode.network.LeNet import LeNet5
from pythoncode.src.pythoncode.config import Config


# ----------------------------
# 1. 设置设备
# ----------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = Config.DEVICE
print(f"Using device: {device}")

# ----------------------------
# 2. 数据加载
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"Test dataset size: {len(test_dataset)}")

# ----------------------------
# 3. 加载模型权重
# ----------------------------
model = LeNet5().to(device)
model_path = './saved_weights/lenet_mnist.pth'

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
print(f"Test Loss: {avg_loss:.4f}")
print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")

# ----------------------------
# 5. 可视化几个预测结果
# ----------------------------
def show_sample_predictions(model, test_dataset, num_samples=8):
    model.eval()
    indices = torch.randint(0, len(test_dataset), (num_samples,))
    images, labels = zip(*[test_dataset[i] for i in indices])
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        img = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"T:{true_label}, P:{pred_label}", 
                         color='green' if true_label == pred_label else 'red')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# 询问是否显示样例
show_sample = input("\n是否显示一些预测样例？(y/n): ").strip().lower()
if show_sample == 'y':
    show_sample_predictions(model, test_dataset)