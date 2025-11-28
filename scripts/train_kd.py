# scripts/train_kd.py

import argparse
import os
import torch

from tqdm.auto import tqdm

from src.data import MNISTConfig, get_loaders, seed_everything, get_device
from src.models import BaselineMLP
from src.train import evaluate, save_checkpoint
from src.kd import kd_loss
from src.utils import pretty_params


def train_kd(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    alpha: float = 0.6,
    temperature: float = 2.0,
    h1_t: int = 256,
    h2_t: int = 256,
    h1_s: int = 128,
    h2_s: int = 128,
    dropout: float = 0.1,
    teacher_ckpt: str = "./outputs/mnist_mlp_best.pt",
    batch_size: int = 128,
    num_workers: int = 0,   # meglio 0 per Colab
    seed: int = 42,
    data_dir: str = "./data",
    out_dir: str = "./outputs_kd",
):
    """
    Knowledge Distillation: teacher -> student (full student MLP).

    Il teacher è un BaselineMLP fissato, lo student è un BaselineMLP più piccolo.
    """
    os.makedirs(out_dir, exist_ok=True)

    # setup
    seed_everything(seed)
    device = get_device()

    # data
    train_loader, test_loader = get_loaders(
        MNISTConfig(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # teacher
    teacher = BaselineMLP(
        hidden_dim=h1_t,
        hidden_dim2=h2_t,
        dropout=dropout,
    ).to(device)
    state = torch.load(teacher_ckpt, map_location="cpu")
    # Compatibilità con checkpoint salvati come {"model_state": ...}
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # student
    student = BaselineMLP(
        hidden_dim=h1_s,
        hidden_dim2=h2_s,
        dropout=dropout,
    ).to(device)
    print("Student:", pretty_params(student))
    print("Teacher:", pretty_params(teacher))

    opt = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # config da salvare nel checkpoint
    cfg = dict(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha,
        temperature=temperature,
        h1_t=h1_t,
        h2_t=h2_t,
        h1_s=h1_s,
        h2_s=h2_s,
        dropout=dropout,
        teacher_ckpt=teacher_ckpt,
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
            save_checkpoint(
                os.path.join(out_dir, "mnist_student_kd_best.pt"),
                {
                    "model_state": student.state_dict(),
                    "acc": best_acc,
                    "cfg": cfg,
                },
            )

    torch.save(
        student.state_dict(),
        os.path.join(out_dir, "mnist_student_kd_final.pt"),
    )
    print(f"KD complete. Best test acc: {best_acc:.2f}%")
    return best_acc


def main():
    """Entry point CLI: parse args e chiama train_kd()."""
    ap = argparse.ArgumentParser(
        description="Knowledge Distillation: teacher -> student (full student)"
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--h1_t", type=int, default=256)
    ap.add_argument("--h2_t", type=int, default=256)
    ap.add_argument("--h1_s", type=int, default=128)
    ap.add_argument("--h2_s", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--teacher_ckpt",
        type=str,
        default="./outputs/mnist_mlp_best.pt",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs_kd")
    args = ap.parse_args()

    train_kd(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        temperature=args.temperature,
        h1_t=args.h1_t,
        h2_t=args.h2_t,
        h1_s=args.h1_s,
        h2_s=args.h2_s,
        dropout=args.dropout,
        teacher_ckpt=args.teacher_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
