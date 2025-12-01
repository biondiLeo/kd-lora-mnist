# scripts/summary_models.py

import torch

from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.lora import copy_baseline_to_lora
from src.utils import summarize_model, layer_table


# ---------------------------------------------------------------------------
# Helper caricamento pesi
# ---------------------------------------------------------------------------

def _load_baseline_state(model, ckpt_path: str):
    """Carica i pesi di un baseline MLP (compatibile con {'model_state': ...})."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    return model


def _load_lora_state_into_model(model, ckpt_path: str, key: str = "lora_state"):
    """
    Carica SOLO i pesi LoRA (matrici A/B) in un modello MNIST_MLP_LoRA.

    I checkpoint LoRA / KD-LoRA sono salvati come:
        {"lora_state": {...}, "acc": ..., "cfg": ...}
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and key in state:
        lora_state = state[key]
    else:
        lora_state = state

    # strict=False perché il modello ha anche i pesi base già inizializzati
    model.load_state_dict(lora_state, strict=False)
    return model


# ---------------------------------------------------------------------------
# Builder dei 4 modelli (CPU, per summary)
# ---------------------------------------------------------------------------

def build_teacher_baseline(
    ckpt_path: str = "./outputs/mnist_mlp_best.pt",
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
) -> torch.nn.Module:
    model = BaselineMLP(hidden_dim=h1, hidden_dim2=h2, dropout=dropout)
    _load_baseline_state(model, ckpt_path)
    return model


def build_kd_student(
    ckpt_path: str = "./outputs_kd/mnist_student_kd_best.pt",
    h1_s: int = 128,
    h2_s: int = 128,
    dropout: float = 0.1,
) -> torch.nn.Module:
    model = BaselineMLP(hidden_dim=h1_s, hidden_dim2=h2_s, dropout=dropout)
    _load_baseline_state(model, ckpt_path)
    return model


def build_lora(
    lora_ckpt_path: str = "./outputs_lora/mnist_lora_best.pt",
    teacher_ckpt_path: str = "./outputs/mnist_mlp_best.pt",
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    rank: int = 8,
    alpha: float = 1.0,
) -> torch.nn.Module:
    # teacher baseline
    teacher = build_teacher_baseline(
        ckpt_path=teacher_ckpt_path,
        h1=h1,
        h2=h2,
        dropout=dropout,
    )
    teacher.eval()

    # modello LoRA
    model = MNIST_MLP_LoRA(
        hidden_dim=h1,
        hidden_dim2=h2,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    # copia pesi base dal teacher e carica le matrici LoRA
    copy_baseline_to_lora(teacher, model)
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
) -> torch.nn.Module:
    teacher = build_teacher_baseline(
        ckpt_path=teacher_ckpt_path,
        h1=h1,
        h2=h2,
        dropout=dropout,
    )
    teacher.eval()

    model = MNIST_MLP_LoRA(
        hidden_dim=h1,
        hidden_dim2=h2,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    copy_baseline_to_lora(teacher, model)
    _load_lora_state_into_model(model, lora_ckpt_path)
    return model


# ---------------------------------------------------------------------------
# Funzioni di summary
# ---------------------------------------------------------------------------

def print_model_summary(title: str, model: torch.nn.Module):
    """
    Stampa:
      - titolo
      - numero di "layer" (tensori di parametri)
      - summary completo (architettura + tabella parametri con tag LoRA/BASE)
    """
    rows = layer_table(model)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Number of parameter tensors (layers): {len(rows)}")

    # questa funzione stampa:
    # - nome classe
    # - architettura (repr del modello)
    # - tabella con: name, shape, #params, trainable, tag (LoRA / BASE / "")
    summarize_model(model)


def summarize_all_models(
    baseline_ckpt: str = "./outputs/mnist_mlp_best.pt",
    kd_ckpt: str = "./outputs_kd/mnist_student_kd_best.pt",
    lora_ckpt: str = "./outputs_lora/mnist_lora_best.pt",
    kd_lora_ckpt: str = "./outputs_kd_lora/mnist_kd_lora_best.pt",
    h1_teacher: int = 256,
    h2_teacher: int = 256,
    h1_student: int = 128,
    h2_student: int = 128,
    dropout: float = 0.1,
    rank: int = 8,
    alpha_lora: float = 1.0,
):
    """
    Costruisce e stampa il summary per:
      - Baseline (Teacher)
      - KD Student
      - LoRA
      - KD-LoRA
    """
    teacher = build_teacher_baseline(
        ckpt_path=baseline_ckpt,
        h1=h1_teacher,
        h2=h2_teacher,
        dropout=dropout,
    )

    kd_student = build_kd_student(
        ckpt_path=kd_ckpt,
        h1_s=h1_student,
        h2_s=h2_student,
        dropout=dropout,
    )

    lora_model = build_lora(
        lora_ckpt_path=lora_ckpt,
        teacher_ckpt_path=baseline_ckpt,
        h1=h1_teacher,
        h2=h2_teacher,
        dropout=dropout,
        rank=rank,
        alpha=alpha_lora,
    )

    kd_lora_model = build_kd_lora(
        lora_ckpt_path=kd_lora_ckpt,
        teacher_ckpt_path=baseline_ckpt,
        h1=h1_teacher,
        h2=h2_teacher,
        dropout=dropout,
        rank=rank,
        alpha=alpha_lora,
    )

    print_model_summary("Baseline (Teacher)", teacher)
    print_model_summary("KD Student", kd_student)
    print_model_summary("LoRA", lora_model)
    print_model_summary("KD-LoRA", kd_lora_model)


# ---------------------------------------------------------------------------
# Uso da riga di comando (opzionale)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summarize_all_models()
