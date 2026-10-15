import os
import shutil
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ["cone", "barrier", "stop_sign"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
DISPLAY_NAMES = {"cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    def clip(self, img_w, img_h):
        return Box(self.cls, min(max(self.x1, 0.0), img_w), min(max(self.y1, 0.0), img_h),
                   min(max(self.x2, 0.0), img_w), min(max(self.y2, 0.0), img_h))

    def to_yolo(self, img_w, img_h):
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        return f"{self.cls} {cx:.6f} {cy:.6f} {self.w / img_w:.6f} {self.h / img_h:.6f}"

    @classmethod
    def from_yolo(cls, line, img_w, img_h):
        c, cx, cy, w, h = line.split()[:5]
        cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
        return cls(int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @classmethod
    def around(cls, cls_id, points):
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return cls(cls_id, min(xs), min(ys), max(xs), max(ys))


def iou(a, b):
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def big_enough(box, min_side=4.0, min_area=64.0):
    return box.w >= min_side and box.h >= min_side and box.w * box.h >= min_area


def label_path(image_path):
    parts = list(Path(image_path).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path, img_w, img_h):
    path = Path(path)
    if not path.exists():
        return []
    return [Box.from_yolo(l, img_w, img_h) for l in path.read_text().splitlines() if l.strip()]


def write_labels(path, boxes, img_w, img_h):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [b.to_yolo(img_w, img_h) for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def link_image(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
