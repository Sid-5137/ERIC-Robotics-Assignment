import argparse
import json
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from detection.labels import CLASS_IDS, Box, big_enough, link_image, write_labels

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{file}"
STOP_SIGN = "stop sign"


def _download(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def _annotations(raw: Path, coco_split: str) -> dict:
    json_path = raw / "annotations" / f"instances_{coco_split}.json"
    if not json_path.exists():
        print("downloading COCO annotations (~250 MB)...")
        zip_path = _download(ANN_URL, raw / "annotations_trainval2017.zip")
        with zipfile.ZipFile(zip_path) as z:
            z.extract(f"annotations/instances_{coco_split}.json", raw)
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def convert(raw: Path, out: Path, coco_split: str, workers: int, min_side: float) -> None:
    data = _annotations(raw, coco_split)
    stop_id = next(c["id"] for c in data["categories"] if c["name"] == STOP_SIGN)
    images = {im["id"]: im for im in data["images"]}

    boxes = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] != stop_id or ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        boxes[ann["image_id"]].append(Box(CLASS_IDS["stop_sign"], x, y, x + w, y + h))

    img_dir = raw / coco_split
    todo = [images[i] for i in boxes if not (img_dir / images[i]["file_name"]).exists()]
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(_download, IMG_URL.format(split=coco_split, file=im["file_name"]),
                            img_dir / im["file_name"]) for im in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"COCO {coco_split} images"):
            f.result()

    split = "train" if coco_split.startswith("train") else "val"
    n_img = n_box = 0
    for img_id, bs in boxes.items():
        im = images[img_id]
        w, h = im["width"], im["height"]
        bs = [b.clip(w, h) for b in bs]
        bs = [b for b in bs if big_enough(b, min_side=min_side)]
        if not bs:
            continue
        stem = f"coco_{Path(im['file_name']).stem}"
        link_image(img_dir / im["file_name"], out / "images" / split / f"{stem}.jpg")
        write_labels(out / "labels" / split / f"{stem}.txt", bs, w, h)
        n_img += 1
        n_box += len(bs)
    print(f"[COCO {coco_split}] {n_img} images, {n_box} stop-sign boxes -> {split}")


def main():
    p = argparse.ArgumentParser(description="COCO 2017 stop signs to YOLO.")
    p.add_argument("--raw", type=Path, default=Path("data/raw/coco"))
    p.add_argument("--out", type=Path, default=Path("datasets/merged"))
    p.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--min-side", type=float, default=8, help="drop stop signs smaller than this (px)")
    a = p.parse_args()
    for s in a.splits:
        convert(a.raw, a.out, s, a.workers, a.min_side)


if __name__ == "__main__":
    main()
