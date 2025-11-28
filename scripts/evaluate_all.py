import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd

from src.data import MNISTConfig, get_loaders, get_device
from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.lora import copy_baseline_to_lora, load_lora_weights
from src.train import evaluate
from src.utils import count_params, summarize_model


def evaluate_kd_alignment(model, teacher, loader, device):
    """
    Calcola la KL-divergence media KL(Student || Teacher) sul test set.
    Misura quanto bene lo student imita le probabilità del teacher.
    """
    model.eval()
    teacher.eval()
    total_kl, n = 0.0, 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            log_probs_student = F.log_softmax(model(x), dim=1)
            probs_teacher = F.softmax(teacher(x), dim=1)
            kl = F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")
            total_kl += kl.item()
            n += 1

    return total_kl / n


def load_model(arch, args, device):
    """
    Costruisce il modello (baseline o LoRA) e carica i pesi dal checkpoint.

    Parameters
    ----------
    arch : str
        "baseline" oppure "lora".
    args : argparse.Namespace o oggetto con gli stessi attributi
        Deve contenere: ckpt, h1, h2, dropout, rank, alpha, teacher_ckpt (per LoRA).
    device : torch.device

    Returns
    -------
    torch.nn.Module
    """
    if arch == "baseline":
        model = BaselineMLP(
            hidden_dim=args.h1,
            hidden_dim2=args.h2,
            dropout=args.dropout,
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
            hidden_dim=args.h1,
            hidden_dim2=args.h2,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        ).to(device)

        # teacher baseline (necessario per ricostruire i pesi base)
        teacher = BaselineMLP(
            hidden_dim=args.h1,
            hidden_dim2=args.h2,
            dropout=args.dropout,
        ).to(device)
        tstate = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(tstate["model_state"])
        teacher.eval()

        # copia pesi base -> LoRA
        copy_baseline_to_lora(teacher, model)

        # carica SOLO i pesi LoRA dal checkpoint
        load_lora_weights(model, args.ckpt)
        return model

    else:
        raise ValueError(f"Unknown arch: {arch}")


def _evaluate_from_args(args):
    """
    Funzione interna: esegue la valutazione dato un oggetto `args`
    (sia che arrivi da argparse, sia che arrivi da evaluate_checkpoint).

    Restituisce un dizionario `results_row` con nomi di colonne espliciti.
    """
    device = get_device()

    # loader di test
    cfg = MNISTConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _, test_loader = get_loaders(cfg)

    # modello
    model = load_model(args.arch, args, device)

    # parametri del modello
    total_params, trainable_params = count_params(model)

    # stampa architettura, se richiesto
    if args.print_model:
        print("\n=== MODEL SUMMARY ===")
        print(summarize_model(model))

    # valutazione standard (loss, accuracy)
    test_loss, test_accuracy = evaluate(model, test_loader, device)

    # tempo di inferenza (facoltativo)
    inference_time_seconds = None
    if args.timed:
        start = time.time()
        # facciamo qualche passaggio per avere un tempo medio stabile
        for _ in range(3):
            evaluate(model, test_loader, device)
        inference_time_seconds = time.time() - start

    # VRAM di picco (solo se su GPU)
    peak_vram_mb = None
    if args.vram and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    # KL(student || teacher) per verificare l'allineamento con il teacher
    kd_kl_divergence = None
    if args.kd_align and args.teacher_ckpt is not None:
        teacher = BaselineMLP(args.h1, args.h2, args.dropout).to(device)
        tstate = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(tstate["model_state"])
        kd_kl_divergence = evaluate_kd_alignment(model, teacher, test_loader, device)

    # riga di risultati con nomi super espliciti
    results_row = dict(
        checkpoint_path=args.ckpt,
        architecture=args.arch,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        trainable_over_total_ratio=trainable_params / total_params,
        accuracy_per_million_trainable_params=test_accuracy
        / (trainable_params / 1e6),
        inference_time_seconds=inference_time_seconds,
        peak_vram_mb=peak_vram_mb,
        kd_kl_divergence=kd_kl_divergence,
    )

    # scrittura su CSV (se richiesto)
    if args.out_csv is not None:
        csv_path = Path(args.out_csv)
        if not csv_path.exists():
            pd.DataFrame([results_row]).to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([results_row])], ignore_index=True)
            df.to_csv(csv_path, index=False)

        print(f"\n[INFO] Risultati aggiunti al file CSV: {csv_path}")

    # stampa riassuntiva a console
    print("\n=== EVALUATION SUMMARY ===")
    for k, v in results_row.items():
        print(f"{k}: {v}")

    return results_row


def evaluate_checkpoint(
    ckpt,
    arch,
    teacher_ckpt=None,
    h1=256,
    h2=256,
    dropout=0.1,
    rank=8,
    alpha=1.0,
    batch_size=256,
    num_workers=0,
    data_dir="./data",
    timed=False,
    vram=False,
    kd_align=False,
    out_csv=None,
    print_model=False,
):
    """
    API comoda da usare nel notebook per valutare un checkpoint.

    Parameters
    ----------
    ckpt : str
        Percorso del checkpoint da valutare.
    arch : str
        "baseline" oppure "lora".
    teacher_ckpt : str or None
        Checkpoint del teacher (necessario per LoRA e per KL alignment).
    h1, h2 : int
        Dimensioni dei due hidden layer dell'MLP.
    dropout : float
        Probabilità di dropout nei layer fully connected.
    rank : int
        Rank delle matrici LoRA (usato solo se arch == "lora").
    alpha : float
        Fattore di scala LoRA (usato solo se arch == "lora").
    batch_size : int
        Batch size per la valutazione.
    num_workers : int
        Numero di worker per il DataLoader.
    data_dir : str
        Directory del dataset MNIST.
    timed : bool
        Se True, misura il tempo di inferenza.
    vram : bool
        Se True, misura la VRAM di picco (se disponibile GPU).
    kd_align : bool
        Se True, calcola la KL-divergence con il teacher (serve teacher_ckpt).
    out_csv : str or None
        Percorso del CSV a cui appendere la riga di risultati. Se None, non salva.
    print_model : bool
        Se True, stampa il summary dell'architettura.

    Returns
    -------
    dict
        Dizionario con tutti i risultati (stessi campi del CSV).
    """

    class Args:
        pass

    args = Args()
    args.ckpt = ckpt
    args.arch = arch
    args.teacher_ckpt = teacher_ckpt
    args.h1 = h1
    args.h2 = h2
    args.dropout = dropout
    args.rank = rank
    args.alpha = alpha
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.data_dir = data_dir
    args.timed = timed
    args.vram = vram
    args.kd_align = kd_align
    args.out_csv = out_csv
    args.print_model = print_model

    return _evaluate_from_args(args)


def main():
    """
    Entrypoint da linea di comando.

    Esempio di uso:
    python scripts/evaluate_all.py \
        --ckpt ./outputs/mnist_mlp_best.pt \
        --arch baseline \
        --h1 256 --h2 256 --dropout 0.1 \
        --batch_size 256 --num_workers 0 \
        --data_dir ./data \
        --timed --vram \
        --kd_align --teacher_ckpt ./outputs/mnist_mlp_best.pt \
        --out_csv ./reports/mnist_results.csv \
        --print_model
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Percorso checkpoint da valutare")
    ap.add_argument(
        "--arch",
        type=str,
        choices=["baseline", "lora"],
        required=True,
        help="Tipo di architettura: 'baseline' o 'lora'",
    )
    ap.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help="Checkpoint del teacher (necessario per LoRA e KL alignment)",
    )
    ap.add_argument("--h1", type=int, default=256, help="Dimensione primo hidden layer")
    ap.add_argument("--h2", type=int, default=256, help="Dimensione secondo hidden layer")
    ap.add_argument("--dropout", type=float, default=0.1, help="Probabilità di dropout")
    ap.add_argument("--rank", type=int, default=8, help="Rank delle matrici LoRA")
    ap.add_argument("--alpha", type=float, default=1.0, help="Fattore di scala LoRA")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size per il test")
    ap.add_argument("--num_workers", type=int, default=0, help="Num. worker DataLoader")
    ap.add_argument("--data_dir", type=str, default="./data", help="Directory dataset MNIST")
    ap.add_argument("--timed", action="store_true", help="Misura tempo di inferenza")
    ap.add_argument("--vram", action="store_true", help="Misura VRAM di picco (se GPU)")
    ap.add_argument(
        "--kd_align",
        action="store_true",
        help="Calcola KL(Student || Teacher) (richiede teacher_ckpt)",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Percorso file CSV in cui salvare/appendere i risultati",
    )
    ap.add_argument(
        "--print_model",
        action="store_true",
        help="Stampa summary dell'architettura del modello",
    )
    args = ap.parse_args()

    _evaluate_from_args(args)


if __name__ == "__main__":
    main()
