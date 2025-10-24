import torch
from torch import nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    # two conv layers (32, 64) with 5×5, ReLU + 2×2 pool, FC512, softmax(10)
    def __init__(self, num_classes: int = 10, c1: int = 32, c2: int = 64, fc: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5, stride=1, padding=2)  # 5×5
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2)  # 5×5
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7*c2, fc)
        self.fc2 = nn.Linear(fc, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28=>14
        x = self.pool(F.relu(self.conv2(x)))  # 14=>7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)