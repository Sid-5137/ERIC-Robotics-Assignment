import argparse
import shutil
from pathlib import Path

FORMATS = {
    "onnx": dict(format="onnx", simplify=True),
    "openvino": dict(format="openvino"),
    "int8": dict(format="openvino", quantize=8),
}


def export(weights, kind, imgsz=640, data="datasets/merged/data.yaml"):
    from ultralytics import YOLO

    path = Path(YOLO(str(weights)).export(imgsz=imgsz, data=str(Path(data).resolve()), **FORMATS[kind]))
    stem = Path(weights).stem
    target = path.with_name(f"{stem}_{kind}_{imgsz}.onnx" if path.is_file() else f"{stem}_{kind}_{imgsz}_openvino_model")
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    path.rename(target)
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export a trained model to ONNX, OpenVINO FP32 or OpenVINO INT8.")
    p.add_argument("weights", type=Path)
    p.add_argument("--to", nargs="+", choices=list(FORMATS), default=["int8"])
    p.add_argument("--imgsz", type=int, nargs="+", default=[640])
    p.add_argument("--data", default="datasets/merged/data.yaml", help="calibration images for INT8")
    a = p.parse_args()
    for kind in a.to:
        for size in a.imgsz:
            print(kind, size, "->", export(a.weights, kind, size, a.data))
