import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from detection.labels import CLASS_IDS, IMG_EXTS, Box, iou, label_path, read_labels, write_labels

COCO_NAMES = {"stop_sign": "stop sign"}
WORLD_PROMPTS = {"cone": "traffic cone", "barrier": "traffic barrier", "stop_sign": "stop sign"}


def load_teacher(weights, classes):
    from ultralytics import YOLO, YOLOWorld

    if "world" in Path(weights).name.lower():
        model = YOLOWorld(weights)
        model.set_classes([WORLD_PROMPTS[c] for c in classes])
        return model, {i: CLASS_IDS[c] for i, c in enumerate(classes)}

    model = YOLO(weights)
    by_name = {n: i for i, n in model.names.items()}
    mapping = {}
    for c in classes:
        if COCO_NAMES.get(c) not in by_name:
            raise SystemExit(f"{weights} cannot detect '{c}', use a YOLO-World teacher instead")
        mapping[by_name[COCO_NAMES[c]]] = CLASS_IDS[c]
    return model, mapping


def predict(model, images, imgsz, conf, device, chunk=16):
    for i in range(0, len(images), chunk):
        yield from model.predict([str(f) for f in images[i:i + chunk]], imgsz=imgsz, conf=conf,
                                 device=device, verbose=False)


def has_class(image, ids):
    lp = label_path(image)
    return lp.exists() and any(l.split()[:1] and l.split()[0] in ids for l in lp.read_text().splitlines())


def teacher(a):
    verify = {}
    for vj in a.verify:
        verify.update(json.loads(vj.read_text()))
    model, mapping = load_teacher(a.teacher, a.classes)
    ours = set(mapping.values())
    added, unlabelled, seen = Counter(), [], 0

    for split in a.splits:
        folder = a.root / "images" / split
        if not folder.exists():
            continue
        images = sorted(f for f in folder.iterdir()
                        if f.suffix.lower() in IMG_EXTS and f.stem.split("_")[0] in a.source)
        seen += len(images)
        if a.rescore:
            images = [f for f in images if has_class(f, {str(i) for i in ours})]

        for image, r in tqdm(zip(images, predict(model, images, a.imgsz, a.conf, a.device)),
                             total=len(images), desc=f"{a.teacher} {split}"):
            h, w = r.orig_shape
            boxes = [b for b in read_labels(label_path(image), w, h) if b.cls not in ours]
            for xyxy, cls in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                if int(cls) not in mapping:
                    continue
                box = Box(mapping[int(cls)], *xyxy)
                if image.stem in verify:
                    human = max((Box(box.cls, *s) for s in verify[image.stem]), key=lambda s: iou(box, s), default=None)
                    if human is None or iou(box, human) < a.verify_iou:
                        continue
                    box = human
                boxes.append(box)
                added[box.cls] += 1
            write_labels(label_path(image), boxes, w, h)
            if not boxes:
                unlabelled.append(image)

    print(f"added boxes per class id: {dict(added)}")
    if a.drop_empty is not None:
        keep = int((seen - len(unlabelled)) * a.drop_empty)
        random.Random(0).shuffle(unlabelled)
        for image in unlabelled[keep:]:
            image.unlink()
            label_path(image).unlink(missing_ok=True)
        print(f"removed {max(0, len(unlabelled) - keep)} unlabelled images, kept {keep} as background")


def square(a):
    stop = str(CLASS_IDS["stop_sign"])
    kept = removed = 0
    for split in ("train", "val"):
        for f in (a.root / "labels" / split).glob("*.txt"):
            if f.stem.split("_")[0] not in a.source:
                continue
            lines, out = f.read_text().splitlines(), []
            for line in lines:
                parts = line.split()
                if parts and parts[0] == stop:
                    w, h = float(parts[3]) * a.img_size[0], float(parts[4]) * a.img_size[1]
                    if h <= 0 or not a.min_aspect <= w / h <= a.max_aspect or min(w, h) < a.min_px:
                        removed += 1
                        continue
                    kept += 1
                out.append(line)
            if len(out) != len(lines):
                f.write_text("\n".join(out) + ("\n" if out else ""))
    print(f"stop signs kept: {kept}, removed: {removed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fill in classes a dataset never labelled, then clean the result.")
    p.add_argument("--root", type=Path, default=Path("datasets/merged"))
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("teacher", help="label missing classes with a bigger pretrained model")
    t.add_argument("--source", nargs="+", required=True, help="image name prefixes, e.g. nuim bdddet")
    t.add_argument("--classes", nargs="+", required=True, choices=list(CLASS_IDS))
    t.add_argument("--teacher", required=True, help="COCO YOLO (stop signs) or a YOLO-World model")
    t.add_argument("--conf", type=float, default=0.5)
    t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--device", default="0")
    t.add_argument("--splits", nargs="+", default=["train", "val"])
    t.add_argument("--verify", type=Path, nargs="*", default=[], help="human sign boxes from 'bdd100k det'")
    t.add_argument("--verify-iou", type=float, default=0.4)
    t.add_argument("--rescore", action="store_true", help="only revisit images that already have these classes")
    t.add_argument("--drop-empty", type=float, help="delete images left unlabelled, keeping this ratio")

    s = sub.add_parser("square", help="drop teacher stop signs whose box is not roughly square")
    s.add_argument("--source", nargs="+", default=["nuim"])
    s.add_argument("--min-aspect", type=float, default=0.7)
    s.add_argument("--max-aspect", type=float, default=1.4)
    s.add_argument("--min-px", type=float, default=10)
    s.add_argument("--img-size", type=int, nargs=2, default=[1600, 900])

    args = p.parse_args()
    teacher(args) if args.cmd == "teacher" else square(args)
