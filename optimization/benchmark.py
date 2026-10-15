import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2

from optimization.export import export

DEVICE_NAMES = {"0": "GPU (CUDA)", "cpu": "CPU", "intel:cpu": "CPU", "intel:gpu": "iGPU (Intel)"}


def size_mb(path):
    path = Path(path)
    files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def save_table(rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def timed(model, images, imgsz, device, warmup=10, runs=100):
    for img in images[:warmup]:
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    total, infer = [], []
    for img in (images * (runs // len(images) + 1))[:runs]:
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, device=device, verbose=False)[0]
        total.append((time.perf_counter() - t0) * 1000)
        infer.append(r.speed["inference"])
    return statistics.median(infer), statistics.median(total)


def variants(weights, extra, sizes, want, data):
    for w in weights:
        for i, sz in enumerate(sizes):
            if "pt_gpu" in want and i == 0:
                yield w.stem, "PyTorch FP32", w, "0", sz
            if "pt_cpu" in want:
                yield w.stem, "PyTorch FP32", w, "cpu", sz
            runs = []
            if "onnx" in want and i == 0:
                runs.append(("ONNX FP32", "onnx", ["cpu"]))
            if "ov_fp32" in want and i == 0:
                runs.append(("OpenVINO FP32", "openvino", ["intel:cpu", "intel:gpu"]))
            if "ov_int8" in want:
                runs.append(("OpenVINO INT8", "int8", ["intel:cpu", "intel:gpu"] if i == 0 else ["intel:cpu"]))
            for name, kind, devices in runs:
                try:
                    path = export(w, kind, sz, data)
                except Exception as e:
                    print(f"[skip] {w.stem} {name} @{sz}: {e}")
                    continue
                for dev in devices:
                    yield w.stem, name, path, dev, sz
    for w in extra:
        for dev in ("0", "cpu"):
            if ("pt_gpu" if dev == "0" else "pt_cpu") in want:
                yield w.stem, "PyTorch FP32", w, dev, sizes[0]


def sweep(a):
    from ultralytics import YOLO

    data = str(Path(a.data).resolve())
    images = sorted(str(f) for f in (Path(a.data).parent / "images" / "val").glob("*.jpg"))[:50]
    rows, accuracy = [], {}
    for tag, variant, path, device, imgsz in variants(a.weights, a.extra, [a.imgsz] + a.sizes, set(a.variants), a.data):
        model = YOLO(str(path), task="detect")
        row = {"model": tag, "variant": variant, "imgsz": imgsz, "device": DEVICE_NAMES.get(device, device),
               "size_mb": round(size_mb(path), 1)}
        if not a.no_val:
            if (str(path), imgsz) not in accuracy:
                m = model.val(data=data, imgsz=imgsz, batch=1, device=device, plots=False, verbose=False)
                accuracy[(str(path), imgsz)] = (round(m.box.map50, 3), round(m.box.map, 3))
            row["mAP50"], row["mAP50_95"] = accuracy[(str(path), imgsz)]
        infer, total = timed(model, images, imgsz, device)
        row.update(inference_ms=round(infer, 1), latency_ms=round(total, 1), fps=round(1000 / total, 1))
        print(row, flush=True)
        rows.append(row)
        save_table(rows, a.out)


def latency(a):
    from ultralytics import YOLO

    frames = [cv2.imread(str(f)) for f in sorted(Path(a.images).glob("*.jpg"))[:a.n_images]]
    models = []
    for spec in a.models:
        label, rest = spec.split("=", 1)
        path, device = rest.split("|") if "|" in rest else (rest, "cpu")
        m = YOLO(path, task="detect")
        for f in frames[:5]:
            m.predict(f, imgsz=a.imgsz, device=device, verbose=False)
        models.append((label, m, device))

    stages = {label: ([], [], []) for label, *_ in models}
    for _ in range(a.rounds):
        for label, m, device in models:
            for f in frames:
                s = m.predict(f, imgsz=a.imgsz, device=device, verbose=False)[0].speed
                for bucket, key in zip(stages[label], ("preprocess", "inference", "postprocess")):
                    bucket.append(s[key])

    rows = []
    for label, (pre, inf, post) in stages.items():
        pre, inf, post = statistics.median(pre), statistics.median(inf), statistics.median(post)
        rows.append({"model": label, "preprocess_ms": round(pre, 2), "inference_ms": round(inf, 2),
                     "postprocess_ms": round(post, 2), "total_ms": round(pre + inf + post, 2),
                     "inference_fps": round(1000 / inf, 1), "total_fps": round(1000 / (pre + inf + post), 1)})
        print(rows[-1], flush=True)
    save_table(rows, a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Speed and accuracy of model variants on CPU, GPU and Intel iGPU.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="export every variant, then measure size, mAP and FPS")
    s.add_argument("--weights", type=Path, nargs="+", required=True)
    s.add_argument("--extra", type=Path, nargs="*", default=[], help="additional .pt files, e.g. pruned models")
    s.add_argument("--variants", nargs="+", default=["pt_gpu", "pt_cpu", "onnx", "ov_fp32", "ov_int8"],
                   choices=["pt_gpu", "pt_cpu", "onnx", "ov_fp32", "ov_int8"])
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--sizes", type=int, nargs="*", default=[], help="extra input sizes, e.g. 480 320")
    s.add_argument("--data", default="datasets/merged/data.yaml")
    s.add_argument("--no-val", action="store_true")
    s.add_argument("--out", type=Path, default=Path("outputs/benchmark.csv"))

    lt = sub.add_parser("latency", help="per-stage latency, models timed in interleaved rounds")
    lt.add_argument("--models", nargs="+", required=True, help='"label=path|device", device: cpu, 0, intel:cpu, intel:gpu')
    lt.add_argument("--images", default="datasets/merged/images/val")
    lt.add_argument("--n-images", type=int, default=20)
    lt.add_argument("--rounds", type=int, default=5)
    lt.add_argument("--imgsz", type=int, default=640)
    lt.add_argument("--out", type=Path, default=Path("outputs/latency.csv"))

    args = p.parse_args()
    sweep(args) if args.cmd == "sweep" else latency(args)
