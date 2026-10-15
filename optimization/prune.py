import argparse
import gzip
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def compressed_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(gzip.compress(buf.getvalue(), compresslevel=6)) / 1e6


def sparsity(model) -> float:
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.numel()
            zeros += (m.weight == 0).sum().item()
    return zeros / max(total, 1)


def prune_model(weights: Path, amount: float, out: Path):
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model = yolo.model
    convs = [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]
    prune.global_unstructured(convs, pruning_method=prune.L1Unstructured, amount=amount)
    for m, name in convs:
        prune.remove(m, name)

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    keep = {k: ckpt.get(k) for k in ("train_args", "date", "version", "license", "docs")}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.half(), "ema": None, "optimizer": None, "epoch": -1, **keep}, out)
    return model.float()


def main():
    p = argparse.ArgumentParser(description="Global L1 unstructured pruning of all Conv2d weights.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--amounts", type=float, nargs="+", default=[0.3, 0.5])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pruned"))
    a = p.parse_args()

    from ultralytics import YOLO

    base = YOLO(str(a.weights)).model
    print(f"original   sparsity {sparsity(base):.1%}  gzip size {compressed_mb(base):.2f} MB")
    for amount in a.amounts:
        out = a.out_dir / f"{a.weights.stem}_pruned{int(amount * 100)}.pt"
        model = prune_model(a.weights, amount, out)
        print(f"pruned {amount:.0%}  sparsity {sparsity(model):.1%}  gzip size {compressed_mb(model):.2f} MB -> {out}")


if __name__ == "__main__":
    main()
