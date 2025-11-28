"""
Models: plain MLP and LoRA-wrapped MLP for MNIST (flattened 1D input).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora import LinearLoRA

class BaselineMLP(nn.Module):
    """Simple MLP for MNIST 1D (flatten 28x28 -> 784)."""
    def __init__(self, input_dim=784, hidden_dim=256, hidden_dim2=256, num_classes=10, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x)

class MNIST_MLP_LoRA(nn.Module):
    """LoRA-wrapped MLP: each Linear adapted with LoRA; base weights frozen."""
    def __init__(self, input_dim=784, hidden_dim=256, hidden_dim2=256, num_classes=10,
                 dropout=0.1, rank=8, alpha=1.0):
        super().__init__()
        self.fc1 = LinearLoRA(nn.Linear(input_dim, hidden_dim), rank=rank, alpha=alpha)
        self.fc2 = LinearLoRA(nn.Linear(hidden_dim, hidden_dim2), rank=rank, alpha=alpha)
        self.fc3 = LinearLoRA(nn.Linear(hidden_dim2, num_classes), rank=rank, alpha=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x)
