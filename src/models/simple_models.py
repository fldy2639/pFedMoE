import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleExtractor(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.adapt = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 8 * 8, feature_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class LocalHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class LinearGate(nn.Module):
    def __init__(self, input_shape, hidden_dim: int):
        super().__init__()
        in_dim = 1
        for d in input_shape:
            in_dim *= d
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def clone_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)
