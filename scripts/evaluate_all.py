import argparse, os, time, torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

from src.data import MNISTConfig, get_loaders, get_device
from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.lora import copy_baseline_to_lora, load_lora_weights
from src.train import evaluate
from src.utils import count_params, summarize_model


def evaluate_kd_alignment(model, teacher, loader, device):
    """KL(Student || Teacher) averaged on test set."""
    model.eval()
    teacher.eval()
    total_kl, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            log_s = F.log_softmax(model(x), dim=1)
            t     = F.softmax(teacher(x), dim=1)
            kl = F.kl_div(log_s, t, reduction="batchmean")
            total_kl += kl.item()
            n += 1
    return total_kl / n


def load_model(arch, args, device):
    """Costruisce modello baseline o LoRA + carica pesi."""
    if arch == "baseline":
        model = BaselineMLP(
            hidden_dim = args.h1,
            hidden_dim2= args.h2,
            dropout    = args.dropout
        ).to(device)

        state = torch.load(args.ckpt, map_location="cpu")
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        return model

    elif arch == "lora":
        # ricrea modello LoRA
        model = MNIST_MLP_LoRA(
            hidden_dim = args.h1,
            hidden_dim2= args.h2,
            rank       = args.rank,
            alpha      = args.alpha,
            dropout    = args.dropout
        ).to(device)

        # teacher baseline (richiesto per ricostruire i pesi base)
        teacher = BaselineMLP(
            hidden_dim = args.h1,
            hidden_dim2= args.h2,
            dropout = args.dropout
        ).to(device)
        tstate = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(tstate["model_state"])
        teacher.eval()

        # copia pesi base -> LoRA
        copy_baseline_to_lora(teacher, model)

        # carica SOLO i pesi LoRA
        load_lora_weights(model, args.ckpt)
        return model

    else:
        raise ValueError(f"Unknown arch: {arch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--arch", type=str, choices=["baseline", "lora"])
    ap.add_argument("--teacher_ckpt", type=str, default=None)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--vram", action="store_true")
    ap.add_argument("--kd_align", action="store_true")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--print_model", action="store_true")
    args = ap.parse_args()

    device = get_device()

    # loader
    cfg = MNISTConfig(
        data_dir = args.data_dir,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    _, test_loader = get_loaders(cfg)

    # modello
    model = load_model(args.arch, args, device)

    # parametri
    total_params, trainable_params = count_params(model)

    # stampa architettura
    if args.print_model:
        print("\n=== MODEL SUMMARY ===")
        print(summarize_model(model))

    # valutazione normale
    loss, acc = evaluate(model, test_loader, device)

    # tempo inferenza
    inf_time = None
    if args.timed:
        start = time.time()
        for _ in range(3):
            evaluate(model, test_loader, device)
        inf_time = time.time() - start

    # GPU VRAM
    peak_vram = None
    if args.vram and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2

    # KL(student || teacher)
    kl = None
    if args.kd_align and args.teacher_ckpt is not None:
        teacher = BaselineMLP(args.h1, args.h2, args.dropout).to(device)
        tstate = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(tstate["model_state"])
        kl = evaluate_kd_alignment(model, teacher, test_loader, device)

    # scrivi CSV
    row = dict(
        ckpt=args.ckpt,
        arch=args.arch,
        acc=acc,
        loss=loss,
        total_params=total_params,
        trainable_params=trainable_params,
        compression_ratio=trainable_params/total_params,
        acc_per_million_trainable=acc/(trainable_params/1e6),
        infer_time=inf_time,
        peak_vram=peak_vram,
        kd_kl=kl
    )

    csv = Path(args.out_csv)
    if not csv.exists():
        pd.DataFrame([row]).to_csv(csv, index=False)
    else:
        df = pd.read_csv(csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv, index=False)

    print("\n=== DONE ===")
    print(f"Appended results to {csv}")


if __name__ == "__main__":
    main()
