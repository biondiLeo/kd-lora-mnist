# src/utils.py

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def pretty_params(model) -> str:
    total, trainable = count_params(model)
    pct = 100.0 * trainable / max(1, total)
    return f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)"

def layer_table(model, include_buffers: bool = False):
    """Return a list of rows with (name, shape, #params, requires_grad) for parameters (and optional buffers)."""
    rows = []
    for name, p in model.named_parameters():
        rows.append((name, list(p.shape), p.numel(), p.requires_grad))
    if include_buffers:
        for name, b in model.named_buffers():
            rows.append((name + " [buffer]", list(b.shape), b.numel(), False))
    return rows

def summarize_model(model, include_buffers: bool = False, max_name: int = 60):
    """Print a readable summary of the model with trainable/frozen and LoRA/BASE tags."""
    print("\n=== MODEL SUMMARY ===")
    print(model.__class__.__name__)
    print(model)
    print("\n--- Parameters ---")
    rows = layer_table(model, include_buffers=include_buffers)
    header = f"{'name':{max_name}}  {'shape':>15}  {'#params':>10}  {'trainable':>9}  {'tag':>8}"
    print(header)
    print("-" * len(header))
    total = 0
    trainable = 0
    for (name, shape, numel, req) in rows:
        total += numel
        if req:
            trainable += numel
        tag = "LoRA" if "lora_" in name else ("BASE" if ".base." in name or name.endswith(".base.weight") or name.endswith(".base.bias") else "")
        dname = (name[:max_name-3] + "...") if len(name) > max_name else name
        print(f"{dname:{max_name}}  {str(shape):>15}  {numel:>10,}  {str(req):>9}  {tag:>8}")
    frozen = total - trainable
    pct = 100.0 * trainable / max(1, total)
    print("-" * len(header))
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,} ({pct:.2f}% trainable)\n")
