# scripts/train_kd_lora.py

import argparse
import os
import torch

from tqdm.auto import tqdm

from src.data import MNISTConfig, get_loaders, seed_everything, get_device
from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.train import evaluate, save_checkpoint, load_state_dict_safely
from src.kd import kd_loss
from src.lora import copy_baseline_to_lora
from src.utils import pretty_params


def train_kd_lora(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    rank: int = 8,
    alpha: float = 0.6,
    temperature: float = 2.0,
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    teacher_ckpt: str = "./outputs/mnist_mlp_best.pt",
    unfreeze_last: bool = False,
    batch_size: int = 128,
    num_workers: int = 0,  # per Colab
    seed: int = 42,
    data_dir: str = "./data",
    out_dir: str = "./outputs_kd_lora",
):
    """
    KD + LoRA: teacher fisso; student è MLP con LoRA.
    """
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(seed)
    device = get_device()
    train_loader, test_loader = get_loaders(
        MNISTConfig(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # teacher
    teacher = BaselineMLP(
        hidden_dim=h1,
        hidden_dim2=h2,
        dropout=dropout,
    ).to(device)
    t_state = torch.load(teacher_ckpt, map_location="cpu")
    load_state_dict_safely(teacher, t_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # student LoRA
    student = MNIST_MLP_LoRA(
        hidden_dim=h1,
        hidden_dim2=h2,
        rank=rank,
        alpha=1.0,   # alpha effettivo è nel LoRA interno; KD usa alpha separato
        dropout=dropout,
    ).to(device)
    copy_baseline_to_lora(teacher, student)

    if unfreeze_last and hasattr(student.fc3, "base"):
        for p in student.fc3.base.parameters():
            p.requires_grad_(True)

    print(pretty_params(student))

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    cfg = dict(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        rank=rank,
        alpha=alpha,
        temperature=temperature,
        h1=h1,
        h2=h2,
        dropout=dropout,
        teacher_ckpt=teacher_ckpt,
        unfreeze_last=unfreeze_last,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        data_dir=data_dir,
        out_dir=out_dir,
    )

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        student.train()
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

            with torch.no_grad():
                t_logits = teacher(x)

            s_logits = student(x)
            loss = kd_loss(
                s_logits,
                t_logits,
                y,
                alpha=alpha,
                temperature=temperature,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pred = s_logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            tr_acc_batch = 100.0 * correct / max(1, total)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{tr_acc_batch:.2f}%",
            )

        tr_loss = running_loss / total
        tr_acc = 100.0 * correct / total
        te_loss, te_acc = evaluate(student, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
            f"test_loss={te_loss:.4f} acc={te_acc:.2f}%"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            lora_state = {
                k: v.cpu()
                for k, v in student.state_dict().items()
                if "lora_" in k
            }
            save_checkpoint(
                os.path.join(out_dir, "mnist_kd_lora_best.pt"),
                {"lora_state": lora_state, "acc": best_acc, "cfg": cfg},
            )

    print(f"KD-LoRA complete. Best test acc: {best_acc:.2f}%")
    return best_acc


def main():
    """Entry point CLI: parse args e chiama train_kd_lora()."""
    ap = argparse.ArgumentParser(
        description="KD + LoRA: teacher fixed; student updates via LoRA"
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--teacher_ckpt",
        type=str,
        default="./outputs/mnist_mlp_best.pt",
    )
    ap.add_argument("--unfreeze_last", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs_kd_lora")
    args = ap.parse_args()

    train_kd_lora(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rank=args.rank,
        alpha=args.alpha,
        temperature=args.temperature,
        h1=args.h1,
        h2=args.h2,
        dropout=args.dropout,
        teacher_ckpt=args.teacher_ckpt,
        unfreeze_last=args.unfreeze_last,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
