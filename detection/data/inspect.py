import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from detection.labels import CLASS_IDS, CLASS_NAMES, IMG_EXTS, label_path, read_labels

COLORS = [(0, 140, 255), (0, 215, 255), (40, 40, 230)]


def stats(root):
    for split in ("train", "val"):
        folder = root / "labels" / split
        if not folder.exists():
            continue
        boxes, images, empty = defaultdict(Counter), Counter(), Counter()
        for f in folder.glob("*.txt"):
            source = f.stem.split("_")[0]
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            images[source] += 1
            empty[source] += not lines
            boxes[source].update(int(l.split()[0]) for l in lines)

        print(f"\n== {split} ==")
        print(f"{'source':<10}{'images':>8}{'bg':>7}" + "".join(f"{c:>11}" for c in CLASS_NAMES))
        total = Counter()
        for source in sorted(images):
            total.update(boxes[source])
            print(f"{source:<10}{images[source]:>8}{empty[source]:>7}"
                  + "".join(f"{boxes[source][i]:>11}" for i in range(len(CLASS_NAMES))))
        print(f"{'TOTAL':<10}{sum(images.values()):>8}{sum(empty.values()):>7}"
              + "".join(f"{total[i]:>11}" for i in range(len(CLASS_NAMES))))


def preview(root, split, sources, only, n, out, seed=0):
    images = [f for f in (root / "images" / split).iterdir()
              if f.suffix.lower() in IMG_EXTS and (not sources or f.stem.split("_")[0] in sources)]
    if only:
        cid = str(CLASS_IDS[only])
        images = [f for f in images if label_path(f).exists()
                  and any(l.split()[:1] == [cid] for l in label_path(f).read_text().splitlines())]
    random.Random(seed).shuffle(images)
    out.mkdir(parents=True, exist_ok=True)
    for f in images[:n]:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        for b in read_labels(label_path(f), w, h):
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), COLORS[b.cls], 2)
            cv2.putText(img, CLASS_NAMES[b.cls], (int(b.x1), max(14, int(b.y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[b.cls], 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{split}_{f.stem}.jpg"), img)
    print(f"wrote {min(n, len(images))} previews to {out} ({len(images)} candidates)")


def sheet(folder, cols, n, tile_w, out):
    files = sorted(Path(folder).glob("*.jpg"))[:n]
    tiles = []
    for f in files:
        img = cv2.imread(str(f))
        tiles.append(cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1]))))
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(files)} images -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Class counts per source, drawn labels and contact sheets.")
    p.add_argument("--root", type=Path, default=Path("datasets/merged"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    pv = sub.add_parser("preview")
    pv.add_argument("--split", default="train")
    pv.add_argument("--source", nargs="*", default=[])
    pv.add_argument("--only", choices=CLASS_NAMES)
    pv.add_argument("--n", type=int, default=24)
    pv.add_argument("--out", type=Path, default=Path("outputs/label_preview"))
    sh = sub.add_parser("sheet")
    sh.add_argument("folder", type=Path)
    sh.add_argument("--cols", type=int, default=4)
    sh.add_argument("--n", type=int, default=12)
    sh.add_argument("--tile-w", type=int, default=480)
    sh.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    if a.cmd == "stats":
        stats(a.root)
    elif a.cmd == "preview":
        preview(a.root, a.split, a.source, a.only, a.n, a.out)
    else:
        sheet(a.folder, a.cols, a.n, a.tile_w, a.out)
