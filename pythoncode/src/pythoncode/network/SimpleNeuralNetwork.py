import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNeuralNetwork, self).__init__()
        
        # 网络结构：输入(784) -> 隐藏层(128, ReLU) -> 输出(10)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Xavier初始化（与C++版本保持一致）
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # 展平输入（如果是28x28图像）
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 第一层：输入 -> 隐藏层 (ReLU激活)
        x = torch.relu(self.fc1(x))
        
        # 第二层：隐藏层 -> 输出层
        x = self.fc2(x)
        
        return x

class SimpleNeuralNetworkG(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNeuralNetworkG, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # 用于保存中间激活
        self.feature_maps = {}

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        self.feature_maps['fc1'] = x.clone()

        x = self.fc2(x)
        self.feature_maps['output'] = x.clone()

        return x