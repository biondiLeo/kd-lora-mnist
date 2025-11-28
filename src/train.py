"""Common training + evaluation utilities."""
from typing import Tuple, Dict, Any
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / total, (100.0 * correct / total)

def train_one_epoch(model, loader, device, optimizer) -> Tuple[float, float]:
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    pbar = tqdm(loader, leave=True, dynamic_ncols=True, desc="train")
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(100.0*correct/total):.2f}%")
    return running_loss / total, (100.0 * correct / total)

def save_checkpoint(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)

def load_state_dict_safely(model, state):
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing = model.load_state_dict(state, strict=False)
    return missing
