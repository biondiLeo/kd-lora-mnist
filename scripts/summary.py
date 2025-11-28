import argparse, torch
from src.models import BaselineMLP, MNIST_MLP_LoRA
from src.lora import copy_baseline_to_lora, load_lora_weights
from src.utils import summarize_model
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Print model summary")
    ap.add_argument("--arch", choices=["baseline","lora"], required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, default=None)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arch == "baseline":
        model = BaselineMLP(args.h1, args.h2, args.dropout)
        state = torch.load(args.ckpt, map_location="cpu")
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

    elif args.arch == "lora":
        model = MNIST_MLP_LoRA(args.h1, args.h2, args.rank, args.alpha, args.dropout)

        # teacher necessaria
        teacher = BaselineMLP(args.h1, args.h2, args.dropout)
        tstate = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(tstate["model_state"])

        copy_baseline_to_lora(teacher, model)
        load_lora_weights(model, args.ckpt)

    model = model.to(device)

    print("=== MODEL SUMMARY ===")
    print(summarize_model(model))


if __name__ == "__main__":
    main()
