import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from detection.labels import CLASS_IDS, Box, big_enough, link_image, write_labels

CATEGORY_MAP = {
    "movable_object.trafficcone": "cone",
    "movable_object.barrier": "barrier",
}


def _load(table_dir: Path, name: str):
    with open(table_dir / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def convert(root: Path, out: Path, split: str, max_images: int, neg_ratio: float,
            cameras: set, seed: int) -> None:
    table_dir = root / f"v1.0-{split}"
    categories = {c["token"]: c["name"] for c in _load(table_dir, "category")}
    sample_data = {sd["token"]: sd for sd in _load(table_dir, "sample_data")}
    calib = {c["token"]: c for c in _load(table_dir, "calibrated_sensor")}

    boxes_per_image = defaultdict(list)
    for ann in _load(table_dir, "object_ann"):
        our_name = CATEGORY_MAP.get(categories[ann["category_token"]])
        if our_name is None:
            continue
        x1, y1, x2, y2 = ann["bbox"]
        boxes_per_image[ann["sample_data_token"]].append(Box(CLASS_IDS[our_name], x1, y1, x2, y2))

    def camera_ok(sd):
        return not cameras or sd["filename"].split("/")[1] in cameras

    positives = [t for t in boxes_per_image if camera_ok(sample_data[t])]
    negatives = [t for t, sd in sample_data.items()
                 if sd["is_key_frame"] and t not in boxes_per_image and camera_ok(sd)]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:max_images] if max_images else positives
    negatives = negatives[: int(len(positives) * neg_ratio)]

    yolo_split = "train" if split == "train" else "val"
    intrinsics, counts, missing = {}, Counter(), 0
    for token in tqdm(positives + negatives, desc=f"nuImages {split}"):
        sd = sample_data[token]
        src = root / sd["filename"]
        if not src.exists():
            missing += 1
            continue
        w, h = sd["width"], sd["height"]
        boxes = [b.clip(w, h) for b in boxes_per_image.get(token, [])]
        boxes = [b for b in boxes if big_enough(b)]
        name = f"nuim_{Path(sd['filename']).stem}"
        link_image(src, out / "images" / yolo_split / f"{name}.jpg")
        write_labels(out / "labels" / yolo_split / f"{name}.txt", boxes, w, h)
        intrinsics[name] = calib[sd["calibrated_sensor_token"]]["camera_intrinsic"]
        counts.update(b.cls for b in boxes)

    intr_path = out / "intrinsics_nuim.json"
    existing = json.loads(intr_path.read_text()) if intr_path.exists() else {}
    existing.update(intrinsics)
    intr_path.write_text(json.dumps(existing))

    print(f"[nuImages {split}] images: {len(positives)} positive + {len(negatives)} background"
          f" | boxes per class id: {dict(counts)} | missing files: {missing}")


def main():
    p = argparse.ArgumentParser(description="nuImages to YOLO (cone, barrier).")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("datasets/merged"))
    p.add_argument("--split", choices=["train", "val", "mini"], default="train")
    p.add_argument("--max-images", type=int, default=0, help="cap on positive images (0 = all)")
    p.add_argument("--neg-ratio", type=float, default=0.1, help="background images per positive")
    p.add_argument("--cameras", nargs="*", default=[], help="e.g. CAM_FRONT CAM_FRONT_LEFT (default: all)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    convert(a.root, a.out, a.split, a.max_images, a.neg_ratio, set(a.cameras), a.seed)


if __name__ == "__main__":
    main()
