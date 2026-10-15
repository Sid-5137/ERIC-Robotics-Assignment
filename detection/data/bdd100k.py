import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from detection.labels import CLASS_IDS, Box, big_enough, link_image, write_labels

W, H = 1280, 720
MIRROR = "dgural/bdd100k"
SEG_TO_OURS = {"traffic cone": "cone", "traffic device": "barrier"}


def load_frames(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["frames"] if isinstance(data, dict) else data


def box_of(label):
    if label.get("box2d"):
        b = label["box2d"]
        return b["x1"], b["y1"], b["x2"], b["y2"]
    points = [v for poly in label.get("poly2d") or [] for v in poly["vertices"]]
    if points:
        b = Box.around(0, points)
        return b.x1, b.y1, b.x2, b.y2


def save(images, out, split, name, boxes, prefix):
    src = images / name
    if not src.exists():
        return False
    stem = f"{prefix}_{Path(name).stem}"
    link_image(src, out / "images" / split / f"{stem}.jpg")
    write_labels(out / "labels" / split / f"{stem}.txt", boxes, W, H)
    return True


def download(a):
    from huggingface_hub import hf_hub_download

    meta = Path(hf_hub_download(MIRROR, "samples.json", repo_type="dataset", local_dir=a.out))
    frames = []
    for s in json.loads(meta.read_text(encoding="utf-8"))["samples"]:
        labels = []
        for det in (s.get("detections") or {}).get("detections", []):
            x, y, w, h = det["bounding_box"]
            labels.append({"category": det["label"],
                           "box2d": {"x1": x * W, "y1": y * H, "x2": (x + w) * W, "y2": (y + h) * H}})
        if any(l["category"] == "traffic sign" and l["box2d"]["y2"] - l["box2d"]["y1"] >= a.min_sign_px for l in labels):
            frames.append((s["filepath"], {"name": Path(s["filepath"]).name, "labels": labels}))
    print(f"{len(frames)} images with a traffic sign >= {a.min_sign_px:.0f} px")

    todo = [fp for fp, _ in frames if not (a.out / fp).exists()]
    with ThreadPoolExecutor(16) as pool:
        jobs = [pool.submit(hf_hub_download, MIRROR, fp, repo_type="dataset", local_dir=a.out) for fp in todo]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="images"):
            job.result()

    random.Random(0).shuffle(frames)
    n_val = int(len(frames) * 0.15)
    for split, part in (("val", frames[:n_val]), ("train", frames[n_val:])):
        (a.out / f"det_{split}.json").write_text(json.dumps([fr for _, fr in part]))
        print(f"{split}: {len(part)} images")


def signs(a):
    frames = load_frames(a.labels)
    random.Random(0).shuffle(frames)
    sign_boxes = {}
    for fr in tqdm(frames, desc="BDD100K signs"):
        found = [box_of(l) for l in fr.get("labels") or [] if l["category"] == "traffic sign"]
        found = [b for b in found if b and b[3] - b[1] >= a.min_sign_px]
        if found and save(a.images, a.out, a.split, fr["name"], [], "bdddet"):
            sign_boxes[f"bdddet_{Path(fr['name']).stem}"] = found
        if a.max_images and len(sign_boxes) >= a.max_images:
            break
    path = a.out / f"bdd_sign_boxes_{a.split}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **sign_boxes}))
    print(f"{len(sign_boxes)} images with traffic signs -> {path}")


def seg(a):
    frames = load_frames(a.labels)
    counts, positives, background = Counter(), 0, []
    for fr in tqdm(frames, desc="BDD100K segmentation"):
        boxes = []
        for label in fr.get("labels") or []:
            target = SEG_TO_OURS.get(label["category"])
            xyxy = box_of(label) if target else None
            if not xyxy:
                continue
            box = Box(CLASS_IDS[target], *xyxy).clip(W, H)
            if target == "cone" and box.h > 0 and box.w / box.h > 1.5:
                continue
            if big_enough(box):
                boxes.append(box)
        if boxes and save(a.images, a.out, a.split, fr["name"], boxes, "bddseg"):
            counts.update(b.cls for b in boxes)
            positives += 1
        elif not boxes:
            background.append(fr["name"])
    random.Random(0).shuffle(background)
    for name in background[: positives // 10]:
        save(a.images, a.out, a.split, name, [], "bddseg")
    print(f"{positives} images with cones/barriers, boxes per class id {dict(counts)}")


def stats(a):
    frames = load_frames(a.labels)
    objects, images, sizes = Counter(), Counter(), defaultdict(list)
    for fr in frames:
        cats = set()
        for label in fr.get("labels") or []:
            objects[label["category"]] += 1
            cats.add(label["category"])
            xyxy = box_of(label)
            if xyxy:
                sizes[label["category"]].append((xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))
        images.update(cats)
    print(f"{len(frames)} images\n{'category':<22}{'objects':>9}{'images':>9}{'med w':>8}{'med h':>8}")
    for cat, n in objects.most_common():
        ws, hs = [s[0] for s in sizes[cat]] or [0], [s[1] for s in sizes[cat]] or [0]
        print(f"{cat:<22}{n:>9}{images[cat]:>9}{statistics.median(ws):>8.0f}{statistics.median(hs):>8.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BDD100K: download, convert to YOLO, inspect.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="images with traffic signs from the Hugging Face mirror")
    d.add_argument("--out", type=Path, default=Path("data/raw/bdd100k_fo"))
    d.add_argument("--min-sign-px", type=float, default=30)

    for name, helptext in (("signs", "detection labels -> stop-sign candidates"),
                           ("seg", "official segmentation polygons -> cone/barrier boxes")):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--labels", type=Path, required=True)
        c.add_argument("--images", type=Path, required=True)
        c.add_argument("--out", type=Path, default=Path("datasets/merged"))
        c.add_argument("--split", choices=["train", "val"], default="train")
        if name == "signs":
            c.add_argument("--min-sign-px", type=float, default=30)
            c.add_argument("--max-images", type=int, default=0)

    s = sub.add_parser("stats", help="category counts in a label file")
    s.add_argument("--labels", type=Path, required=True)

    a = p.parse_args()
    {"download": download, "signs": signs, "seg": seg, "stats": stats}[a.cmd](a)
