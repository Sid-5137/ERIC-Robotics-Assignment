import argparse
import sys
from pathlib import Path

import yaml

from detection.labels import CLASS_NAMES


def write_train_list(root: Path, oversample: dict) -> Path:
    repeat_for = {str(CLASS_NAMES.index(c)): k for c, k in oversample.items()}
    lines, extra = [], 0
    for img in sorted((root / "images" / "train").glob("*.jpg")):
        lbl = root / "labels" / "train" / f"{img.stem}.txt"
        classes = {l.split()[0] for l in lbl.read_text().splitlines() if l.strip()} if lbl.exists() else set()
        k = max([repeat_for.get(c, 1) for c in classes] or [1])
        lines += [img.as_posix()] * k
        extra += k - 1
    out = root / "train_oversampled.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"train list: {len(lines)} entries ({extra} extra copies for {oversample})")
    return out


def write_data_yaml(dataset_root: Path, oversample: dict = None) -> Path:
    root = dataset_root.resolve()
    train = write_train_list(root, oversample).as_posix() if oversample else "images/train"
    data = {"path": str(root), "train": train, "val": "images/val",
            "names": dict(enumerate(CLASS_NAMES))}
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def main():
    p = argparse.ArgumentParser(description="Fine-tune a COCO-pretrained YOLO on the merged dataset.")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--resume", action="store_true", help="resume the run named in the config")
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. epochs=30 batch=8")
    p.add_argument("--model")
    p.add_argument("--name")
    a = p.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    for kv in a.overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if a.model:
        cfg["model"] = a.model
    if a.name:
        cfg["name"] = a.name

    from ultralytics import YOLO

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    if a.resume:
        candidates = sorted(Path("runs").rglob(f"{cfg['name']}/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(f"nothing to resume: no {cfg['name']}/weights/last.pt under runs/")
        last = candidates[-1]
        print(f"resuming {last}")
        YOLO(str(last)).train(resume=True)
        return

    data_yaml = write_data_yaml(Path(cfg.pop("dataset_root")), cfg.pop("oversample", None))
    model = YOLO(cfg.pop("model"))
    model.train(data=str(data_yaml), exist_ok=True, **cfg)


if __name__ == "__main__":
    main()
