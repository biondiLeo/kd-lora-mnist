# scripts/train_baseline.py

import argparse, os
import torch

from src.data import MNISTConfig, get_loaders, seed_everything, get_device
from src.models import BaselineMLP
from src.train import train_one_epoch, evaluate, save_checkpoint
from src.utils import pretty_params


def train_baseline(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    data_dir: str = "./data",
    out_dir: str = "./outputs",
):
    """
    Training baseline MLP su MNIST.
    Può essere chiamata sia da notebook che da riga di comando (via main()).
    """
    os.makedirs(out_dir, exist_ok=True)

    # setup
    seed_everything(seed)
    device = get_device()
    train_loader, test_loader = get_loaders(
        MNISTConfig(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # modello
    model = BaselineMLP(hidden_dim=h1, hidden_dim2=h2, dropout=dropout).to(device)
    print(pretty_params(model))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # per salvare la config dentro il checkpoint
    cfg = dict(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        h1=h1,
        h2=h2,
        dropout=dropout,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        data_dir=data_dir,
        out_dir=out_dir,
    )

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
            f"test_loss={te_loss:.4f} acc={te_acc:.2f}%"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(
                os.path.join(out_dir, "mnist_mlp_best.pt"),
                {"model_state": model.state_dict(), "acc": best_acc, "cfg": cfg},
            )

    torch.save(model.state_dict(), os.path.join(out_dir, "mnist_mlp_final.pt"))
    print(f"Training complete. Best test acc: {best_acc:.2f}%")


def main():
    """Entry point da riga di comando: parsare argomenti e chiamare train_baseline()."""
    ap = argparse.ArgumentParser(description="Train MNIST baseline MLP")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    args = ap.parse_args()

    train_baseline(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        h1=args.h1,
        h2=args.h2,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
