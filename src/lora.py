"""
LoRA modules and helpers.
"""
import math
import torch
import torch.nn as nn

class LinearLoRA(nn.Module):
    """
    Wrap an nn.Linear with low-rank adapters A (r x in) and B (out x r).
    Base weights are frozen; only A and B are trainable.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.zeros((rank, base_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.alpha / max(1, self.rank)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        # (out x r) @ (r x in) @ (in x batch) -> (out x batch) -> T -> (batch x out)
        lora_out = (self.lora_B @ self.lora_A) @ x.T
        lora_out = lora_out.T * self.scaling
        return base_out + lora_out

def copy_baseline_to_lora(baseline, lora_model) -> None:
    """
    Copy baseline Linear weights/bias into the frozen 'base' layers
    of a LoRA-wrapped MLP. Expects attributes fc1.base, fc2.base, fc3.base.
    """
    with torch.no_grad():
        lora_model.fc1.base.weight.copy_(baseline.fc1.weight)
        if baseline.fc1.bias is not None and lora_model.fc1.base.bias is not None:
            lora_model.fc1.base.bias.copy_(baseline.fc1.bias)

        lora_model.fc2.base.weight.copy_(baseline.fc2.weight)
        if baseline.fc2.bias is not None and lora_model.fc2.base.bias is not None:
            lora_model.fc2.base.bias.copy_(baseline.fc2.bias)

        lora_model.fc3.base.weight.copy_(baseline.fc3.weight)
        if baseline.fc3.bias is not None and lora_model.fc3.base.bias is not None:
            lora_model.fc3.base.bias.copy_(baseline.fc3.bias)
