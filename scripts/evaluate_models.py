# scripts/evaluate_models.py
"""
Utility di valutazione e confronto per:
- Baseline (teacher)
- KD Student
- LoRA
- KD-LoRA

Pensato per essere:
- importato da notebook (API Python)
- opzionalmente usato anche da riga di comando.
"""

from typing import Callable, Dict, Any, List, Optional
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import pandas as pd

from src.data import MNISTConfig, get_loaders, get_device
from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.utils import count_params
from src.train import evaluate
from src.lora import copy_baseline_to_lora


# ---------------------------------------------------------------------------
# Helper: data & KL
# ---------------------------------------------------------------------------

def get_test_loader(
    batch_size: int = 256,
    num_workers: int = 0,
    data_dir: str = "./data",
):
    """
    Crea un DataLoader di test comune a tutti i modelli.
    """
    cfg = MNISTConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    _, test_loader = get_loaders(cfg)
    return test_loader


def compute_kl_student_teacher(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader,
    device: torch.device,
) -> float:
    """
    KL(Student || Teacher) media sul test set.
    Serve per vedere quanto lo student imita le probabilità del teacher.
    """
    student.eval()
    teacher.eval()
    total_kl, n = 0.0, 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            log_s = F.log_softmax(student(x), dim=1)
            t = F.softmax(teacher(x), dim=1)
            kl = F.kl_div(log_s, t, reduction="batchmean")
            total_kl += kl.item()
            n += 1

    return total_kl / max(1, n)


# ---------------------------------------------------------------------------
# Builder dei 4 modelli (ricostruiti dai checkpoint)
# ---------------------------------------------------------------------------

def build_teacher_baseline(
    ckpt_path: str = "./outputs/mnist_mlp_best.pt",
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Costruisce il modello BaselineMLP (teacher) e carica i pesi dal checkpoint.
    """
    if device is None:
        device = get_device()

    model = BaselineMLP(hidden_dim=h1, hidden_dim2=h2, dropout=dropout)
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model_state", state)
    model.load_state_dict(state)
    return model.to(device)


def build_kd_student(
    ckpt_path: str = "./outputs_kd/mnist_student_kd_best.pt",
    h1_s: int = 128,
    h2_s: int = 128,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Costruisce lo student KD (BaselineMLP più piccolo) e carica i pesi.
    """
    if device is None:
        device = get_device()

    model = BaselineMLP(hidden_dim=h1_s, hidden_dim2=h2_s, dropout=dropout)
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model_state", state)
    model.load_state_dict(state)
    return model.to(device)


def _load_lora_state_into_model(
    model: torch.nn.Module,
    ckpt_path: str,
    key: str = "lora_state",
):
    """
    Carica SOLO i pesi LoRA (matrici A/B) in un modello MNIST_MLP_LoRA.

    I checkpoint LoRA e KD-LoRA sono salvati come:
        {"lora_state": {...}, "acc": ..., "cfg": ...}
    nei rispettivi script di training.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and key in state:
        lora_state = state[key]
    else:
        lora_state = state

    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # Si potrebbe loggare, ma per l'uso normale non è necessario.
    return missing, unexpected


def build_lora(
    lora_ckpt_path: str = "./outputs_lora/mnist_lora_best.pt",
    teacher_ckpt_path: str = "./outputs/mnist_mlp_best.pt",
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    rank: int = 8,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Ricostruisce il modello LoRA:
    - crea il teacher baseline
    - copia i suoi pesi base nel modello LoRA
    - carica SOLO i pesi LoRA dal checkpoint.
    """
    if device is None:
        device = get_device()

    # Teacher baseline
    teacher = build_teacher_baseline(
        ckpt_path=teacher_ckpt_path,
        h1=h1,
        h2=h2,
        dropout=dropout,
        device=device,
    )
    teacher.eval()

    # Modello LoRA
    model = MNIST_MLP_LoRA(
        hidden_dim=h1,
        hidden_dim2=h2,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    ).to(device)

    # Copia pesi base dal teacher
    copy_baseline_to_lora(teacher, model)

    # Applica pesi LoRA
    _load_lora_state_into_model(model, lora_ckpt_path)

    return model


def build_kd_lora(
    lora_ckpt_path: str = "./outputs_kd_lora/mnist_kd_lora_best.pt",
    teacher_ckpt_path: str = "./outputs/mnist_mlp_best.pt",
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    rank: int = 8,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Ricostruisce il modello KD-LoRA:
    stessa logica di build_lora, ma usando il checkpoint KD-LoRA.
    """
    if device is None:
        device = get_device()

    teacher = build_teacher_baseline(
        ckpt_path=teacher_ckpt_path,
        h1=h1,
        h2=h2,
        dropout=dropout,
        device=device,
    )
    teacher.eval()

    model = MNIST_MLP_LoRA(
        hidden_dim=h1,
        hidden_dim2=h2,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    ).to(device)

    copy_baseline_to_lora(teacher, model)
    _load_lora_state_into_model(model, lora_ckpt_path)
    return model


# ---------------------------------------------------------------------------
# Valutazione di un singolo modello
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    build_fn: Callable[[], torch.nn.Module],
    test_loader,
    device: Optional[torch.device] = None,
    teacher_build_fn: Optional[Callable[[], torch.nn.Module]] = None,
) -> Dict[str, Any]:
    """
    Valuta un modello su MNIST e restituisce una riga di risultati.

    Parameters
    ----------
    model_name : str
        Nome leggibile del modello (es: "Baseline (Teacher)", "LoRA", ...).
    build_fn : Callable
        Funzione che costruisce e restituisce il modello già caricato da checkpoint.
    test_loader : DataLoader
        Loader del test set.
    device : torch.device or None
        Device su cui eseguire. Se None, usa get_device().
    teacher_build_fn : Callable or None
        Se fornita, verrà usata per costruire il teacher e calcolare KL(Student || Teacher).

    Returns
    -------
    Dict[str, Any]
        Dizionario con tutte le metriche principali (pronto per DataFrame).
    """
    if device is None:
        device = get_device()

    model = build_fn().to(device)
    model.eval()

    # conta parametri
    total_params, trainable_params = count_params(model)

    # valutazione standard
    t0 = time.time()
    test_loss, test_acc = evaluate(model, test_loader, device)
    infer_time = time.time() - t0

    results = dict(
        model_name=model_name,
        test_accuracy=test_acc,
        test_loss=test_loss,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        trainable_over_total_ratio=trainable_params / max(1, total_params),
        accuracy_per_million_trainable_params=test_acc / (trainable_params / 1e6),
        inference_time_seconds=infer_time,
    )

    # opzionale: allineamento KD con il teacher
    if teacher_build_fn is not None:
        teacher = teacher_build_fn().to(device)
        kd_kl = compute_kl_student_teacher(model, teacher, test_loader, device)
        results["kd_kl_divergence"] = kd_kl

    return results


# ---------------------------------------------------------------------------
# Valutazione "standard" di tutti e 4 i modelli
# ---------------------------------------------------------------------------

def evaluate_all_models(
    baseline_ckpt: str = "./outputs/mnist_mlp_best.pt",
    kd_ckpt: str = "./outputs_kd/mnist_student_kd_best.pt",
    lora_ckpt: str = "./outputs_lora/mnist_lora_best.pt",
    kd_lora_ckpt: str = "./outputs_kd_lora/mnist_kd_lora_best.pt",
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 0,
    h1_teacher: int = 256,
    h2_teacher: int = 256,
    h1_student: int = 128,
    h2_student: int = 128,
    dropout: float = 0.1,
    rank: int = 8,
    alpha_lora: float = 1.0,
    save_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Valuta i 4 modelli "standard" (Baseline, KD, LoRA, KD-LoRA) e restituisce un DataFrame.

    È la funzione da chiamare nel notebook per avere la tabella riassuntiva.
    """
    device = get_device()
    test_loader = get_test_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
    )

    rows: List[Dict[str, Any]] = []

    # 1) Baseline (Teacher)
    rows.append(
        evaluate_model(
            model_name="Baseline (Teacher)",
            build_fn=lambda: build_teacher_baseline(
                ckpt_path=baseline_ckpt,
                h1=h1_teacher,
                h2=h2_teacher,
                dropout=dropout,
                device=device,
            ),
            test_loader=test_loader,
            device=device,
            teacher_build_fn=None,  # il teacher è se stesso
        )
    )

    # 2) KD Student
    rows.append(
        evaluate_model(
            model_name="KD Student",
            build_fn=lambda: build_kd_student(
                ckpt_path=kd_ckpt,
                h1_s=h1_student,
                h2_s=h2_student,
                dropout=dropout,
                device=device,
            ),
            test_loader=test_loader,
            device=device,
            teacher_build_fn=lambda: build_teacher_baseline(
                ckpt_path=baseline_ckpt,
                h1=h1_teacher,
                h2=h2_teacher,
                dropout=dropout,
                device=device,
            ),
        )
    )

    # 3) LoRA
    rows.append(
        evaluate_model(
            model_name="LoRA",
            build_fn=lambda: build_lora(
                lora_ckpt_path=lora_ckpt,
                teacher_ckpt_path=baseline_ckpt,
                h1=h1_teacher,
                h2=h2_teacher,
                dropout=dropout,
                rank=rank,
                alpha=alpha_lora,
                device=device,
            ),
            test_loader=test_loader,
            device=device,
            teacher_build_fn=lambda: build_teacher_baseline(
                ckpt_path=baseline_ckpt,
                h1=h1_teacher,
                h2=h2_teacher,
                dropout=dropout,
                device=device,
            ),
        )
    )

    # 4) KD-LoRA
    rows.append(
        evaluate_model(
            model_name="KD-LoRA",
            build_fn=lambda: build_kd_lora(
                lora_ckpt_path=kd_lora_ckpt,
                teacher_ckpt_path=baseline_ckpt,
                h1=h1_teacher,
                h2=h2_teacher,
                dropout=dropout,
                rank=rank,
                alpha=alpha_lora,
                device=device,
            ),
            test_loader=test_loader,
            device=device,
            teacher_build_fn=lambda: build_teacher_baseline(
                ckpt_path=baseline_ckpt,
                h1=h1_teacher,
                h2=h2_teacher,
                dropout=dropout,
                device=device,
            ),
        )
    )

    df = pd.DataFrame(rows)

    if save_csv_path is not None:
        save_path = Path(save_csv_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)  # <--- crea cartella se non esiste
        df.to_csv(save_path, index=False)

    return df


# ---------------------------------------------------------------------------
# (Facoltativo) entrypoint CLI minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Uso tipico da linea di comando, giusto per comodità:
    # python scripts/evaluate_models.py
    df = evaluate_all_models()
    print(df)
