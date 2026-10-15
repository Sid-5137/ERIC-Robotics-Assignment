import argparse
import csv
from pathlib import Path


def per_class(models, data, imgsz, device, out):
    from ultralytics import YOLO

    rows = []
    for spec in models:
        label, weights = spec.split("=", 1)
        r = YOLO(weights).val(data=data, imgsz=imgsz, batch=8, device=device, plots=True,
                              project=str(out.resolve()), name=label, exist_ok=True, verbose=False)
        row = {"model": label, "mAP50": r.box.map50, "mAP50-95": r.box.map}
        for i, name in r.names.items():
            row[f"{name} AP50"], row[f"{name} AP50-95"], row[f"{name} R"] = r.box.ap50[i], r.box.ap[i], r.box.r[i]
        rows.append(row)
        print(label, {k: round(v, 3) for k, v in row.items() if k != "model"}, flush=True)

    keys = list(rows[0])
    table = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    table += ["| " + " | ".join(r[k] if isinstance(r[k], str) else f"{r[k]:.3f}" for k in keys) + " |" for r in rows]
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_class.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


def curves(runs, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
               ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 3.4))
    for spec in runs:
        label, path = spec.split("=", 1)
        with open(path) as f:
            rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        for ax, (key, _) in zip(axes, metrics):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", markersize=2.5, label=label)
    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validation metrics and training curves.")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("val", help="per-class AP of one or more models")
    v.add_argument("--models", nargs="+", required=True, help="label=path/to/weights.pt")
    v.add_argument("--data", default="datasets/merged/data.yaml")
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--device", default="0")
    v.add_argument("--out", type=Path, default=Path("outputs/eval"))
    c = sub.add_parser("curves", help="plot mAP / precision / recall per epoch")
    c.add_argument("--runs", nargs="+", required=True, help="label=path/to/results.csv")
    c.add_argument("--out", type=Path, default=Path("assets/training_curves.png"))
    a = p.parse_args()

    if a.cmd == "val":
        per_class(a.models, a.data, a.imgsz, a.device, a.out)
    else:
        curves(a.runs, a.out)
