import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[64, 64, 32], output_size=10):
        super(MLPNetwork, self).__init__()
        
        # 创建多层网络
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, output_size)
        
        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 为所有线性层进行Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

class MLPNetworkG(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[64, 64, 32], output_size=10):
        super(MLPNetworkG, self).__init__()
        
        # 创建多层网络
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, output_size)
        
        # 初始化
        self._initialize_weights()
        
        # 用于保存中间激活
        self.feature_maps = {}

    def _initialize_weights(self):
        # 为所有线性层进行Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 保存输入
        self.feature_maps['input'] = x.clone()
        
        # 通过隐藏层
        x = self.features(x)
        self.feature_maps['hidden'] = x.clone()
        
        # 输出层
        x = self.classifier(x)
        self.feature_maps['output'] = x.clone()
        
        return x