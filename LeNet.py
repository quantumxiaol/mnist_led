import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5G(nn.Module):
    def __init__(self):
        super(LeNet5G, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 用于保存中间激活
        self.feature_maps = {}

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        self.feature_maps['conv1'] = x.clone()

        x = torch.max_pool2d(x, 2)
        self.feature_maps['pool1'] = x.clone()

        x = torch.relu(self.conv2(x))
        self.feature_maps['conv2'] = x.clone()

        x = torch.max_pool2d(x, 2)
        self.feature_maps['pool2'] = x.clone()

        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        self.feature_maps['fc1'] = x.clone()

        x = torch.relu(self.fc2(x))
        self.feature_maps['fc2'] = x.clone()

        x = self.fc3(x)
        self.feature_maps['output'] = x.clone()
        return x