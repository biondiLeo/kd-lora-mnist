"""
Data and runtime utilities for MNIST experiments.
"""
from dataclasses import dataclass
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer_mps: bool = True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@dataclass
class MNISTConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081

def get_loaders(cfg: MNISTConfig):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.normalize_mean,), (cfg.normalize_std,)),
    ])
    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, transform=tfm, download=True)
    test_ds  = datasets.MNIST(root=cfg.data_dir, train=False, transform=tfm, download=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader
