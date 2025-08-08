import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from config import Config
from LeNet import LeNet5


# ----------------------------
# 1. Hyperparameters
# ----------------------------
batch_size = 64
learning_rate = 0.001
num_epochs = 20
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = Config.DEVICE

# ----------------------------
# 2. Data Loading (with auto-download)
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Check if data exists; download only if not
data_dir = './data'
train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# ----------------------------
# 3. Initialize Model, Loss, Optimizer
# ----------------------------
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# 4. Training Loop
# ----------------------------
model.train()
print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')

training_end_time = time.time()
total_training_time = training_end_time - start_time
print('Training complete.')

# ----------------------------
# 5. Save Model Weights
# ----------------------------
weights_dir = './saved_weights'
os.makedirs(weights_dir, exist_ok=True)
model_path = os.path.join(weights_dir, 'lenet_mnist.pth')
torch.save(model.state_dict(), model_path)
print(f'Model weights saved to {model_path}')

# ----------------------------
# 6. Evaluation on Test Set
# ----------------------------
model.eval()
correct = 0
total = 0
test_loss = 0.0

print("Evaluating model on test set...")
eval_start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

eval_end_time = time.time()
eval_duration = eval_end_time - eval_start_time

accuracy = 100 * correct / total
avg_test_loss = test_loss / len(test_loader)

print(f'Evaluation complete. Time: {eval_duration:.2f}s')
print(f'Test Loss: {avg_test_loss:.4f}')
print(f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')

print("\n=== Training Summary ===")
print(f"Total Epochs: {num_epochs}")
print(f"Total Training Time: {total_training_time:.2f}s")
print(f"Average Time per Epoch: {total_training_time / num_epochs:.2f}s")
print(f"Total Evaluation Time: {eval_duration:.2f}s")
print(f"Final Test Accuracy: {accuracy:.2f}%")
print(f"Model saved to: {model_path}")