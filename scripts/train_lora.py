# scripts/train_lora.py

import argparse
import os
import torch

from tqdm.auto import tqdm  # meglio per notebook/colab

from src.data import MNISTConfig, get_loaders, seed_everything, get_device
from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.train import evaluate, save_checkpoint, load_state_dict_safely
from src.lora import copy_baseline_to_lora
from src.utils import pretty_params


def train_lora(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    rank: int = 8,
    alpha: float = 1.0,
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    from_baseline: bool = False,
    base_ckpt: str = "./outputs/mnist_mlp_best.pt",
    unfreeze_last: bool = False,
    batch_size: int = 128,
    num_workers: int = 0,   
    seed: int = 42,
    data_dir: str = "./data",
    out_dir: str = "./outputs_lora",
):
    """
    LoRA fine-tuning su MNIST.

    Può partire:
      - da pesi casuali (from_baseline=False)
      - copiando i pesi di un baseline (from_baseline=True, base_ckpt=...)

    Può opzionalmente sbloccare l'ultimo layer FC3 (unfreeze_last=True).
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

    # modello LoRA
    model = MNIST_MLP_LoRA(
        hidden_dim=h1,
        hidden_dim2=h2,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    ).to(device)

    # opzionale: inizializza dalle weights del baseline
    if from_baseline:
        baseline = BaselineMLP(hidden_dim=h1, hidden_dim2=h2, dropout=dropout).to(device)
        state = torch.load(base_ckpt, map_location="cpu")
        load_state_dict_safely(baseline, state)
        baseline.eval()
        copy_baseline_to_lora(baseline, model)

    # opzionale: sblocca l'ultimo layer fully-connected
    if unfreeze_last and hasattr(model.fc3, "base"):
        for p in model.fc3.base.parameters():
            p.requires_grad_(True)

    print(pretty_params(model))

    # ottimizzatore solo sui parametri allenabili
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # cfg da salvare nel checkpoint
    cfg = dict(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        rank=rank,
        alpha=alpha,
        h1=h1,
        h2=h2,
        dropout=dropout,
        from_baseline=from_baseline,
        base_ckpt=base_ckpt,
        unfreeze_last=unfreeze_last,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        data_dir=data_dir,
        out_dir=out_dir,
    )

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2,
        )

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            tr_acc_batch = 100.0 * correct / max(1, total)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{tr_acc_batch:.2f}%",
            )

        tr_loss = running_loss / total
        tr_acc = 100.0 * correct / total
        te_loss, te_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
            f"test_loss={te_loss:.4f} acc={te_acc:.2f}%"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            # salviamo SOLO i pesi LoRA (come prima)
            lora_state = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if "lora_" in k
            }
            save_checkpoint(
                os.path.join(out_dir, "mnist_lora_best.pt"),
                {"lora_state": lora_state, "acc": best_acc, "cfg": cfg},
            )

    print(f"LoRA complete. Best test acc: {best_acc:.2f}%")
    return best_acc


def main():
    """Entry point da linea di comando."""
    ap = argparse.ArgumentParser(
        description="LoRA fine-tuning (from scratch or from baseline)"
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--from_baseline",
        action="store_true",
        help="initialize LoRA bases from a pretrained baseline",
    )
    ap.add_argument(
        "--base_ckpt",
        type=str,
        default="./outputs/mnist_mlp_best.pt",
    )
    ap.add_argument("--unfreeze_last", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs_lora")
    args = ap.parse_args()

    train_lora(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rank=args.rank,
        alpha=args.alpha,
        h1=args.h1,
        h2=args.h2,
        dropout=args.dropout,
        from_baseline=args.from_baseline,
        base_ckpt=args.base_ckpt,
        unfreeze_last=args.unfreeze_last,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
