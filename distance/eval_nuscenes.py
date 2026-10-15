import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detection.labels import Box, iou
from distance.estimator import estimate_distance, load_priors
from distance.nuscenes import NuScenes

BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, math.inf)]
METHODS = ("size", "ground", "fused")


def summarize(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = []
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for method in METHODS:
            pairs = [(r[method], r["gt"]) for r in rs if r.get(method) is not None]
            if not pairs:
                continue
            abs_err = [abs(e - t) for e, t in pairs]
            rel_err = [abs(e - t) / t for e, t in pairs]
            out.append({key: g, "method": method, "n": len(pairs),
                        "MAE_m": round(statistics.mean(abs_err), 2),
                        "median_abs_m": round(statistics.median(abs_err), 2),
                        "AbsRel_%": round(100 * statistics.mean(rel_err), 1),
                        "RMSE_m": round(math.sqrt(statistics.mean(e * e for e in abs_err)), 2)})
    return out


def to_markdown(rows):
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    return "\n".join(lines + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows])


def match(preds, objects, min_iou):
    pairs, unmatched = [], list(objects)
    for xyxy, name, _ in sorted(preds, key=lambda p: -p[2]):
        pb = Box(0, *xyxy)
        cands = [o for o in unmatched if o["class"] == name]
        best = max(cands, key=lambda o: iou(pb, o["box"]), default=None)
        if best is not None and iou(pb, best["box"]) >= min_iou:
            unmatched.remove(best)
            pairs.append((name, pb, best))
    return pairs


def draw(img, pairs, cam, priors):
    for name, b, o in pairs:
        est = estimate_distance((b.x1, b.y1, b.x2, b.y2), name, cam, priors, "size")
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 255), 2)
        text = f"{name} {est.distance_m:.1f}m (gt {o['gt_distance']:.1f}m)" if est else name
        cv2.putText(img, text, (int(b.x1), max(14, int(b.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description="Distance error against nuScenes LiDAR ground truth.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--min-visibility", type=int, default=3)
    p.add_argument("--weights", help="evaluate the detector instead of projected ground-truth boxes")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--fit-priors", action="store_true")
    p.add_argument("--save-vis", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/distance_eval"))
    a = p.parse_args()

    nusc = NuScenes(a.root, a.version)
    frames = [(f, nusc.objects(f, a.min_visibility)) for f in nusc.key_frames(a.camera)]
    frames = [(f, objs) for f, objs in frames if objs]
    n_gt = sum(len(objs) for _, objs in frames)
    print(f"{len(frames)} {a.camera} key frames, {n_gt} cones/barriers")

    if a.fit_priors:
        heights = defaultdict(list)
        for _, objs in frames:
            for o in objs:
                heights[o["class"]].append(o["gt_height"])
        for c, hs in heights.items():
            print(f"{c:<8} n={len(hs):<5} median height {statistics.median(hs):.2f} m "
                  f"(IQR {np.percentile(hs, 25):.2f}-{np.percentile(hs, 75):.2f})")

    priors = load_priors()
    model = None
    if a.weights:
        from ultralytics import YOLO
        model = YOLO(a.weights)

    rows = []
    vis_idx = set(np.linspace(0, len(frames) - 1, a.save_vis).astype(int).tolist()) if a.save_vis else set()
    a.out.mkdir(parents=True, exist_ok=True)
    for i, (frame, objs) in enumerate(frames):
        cam = frame.intrinsics()
        if model is None:
            pairs = [(o["class"], o["box"], o) for o in objs]
        else:
            r = model.predict(str(frame.path), conf=a.conf, device=a.device, verbose=False)[0]
            preds = [(xyxy, r.names[int(c)], s) for xyxy, c, s in
                     zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            pairs = match(preds, objs, a.match_iou)

        for name, b, o in pairs:
            row = {"image": frame.path.name, "class": name, "gt": round(o["gt_distance"], 2)}
            for method in METHODS:
                est = estimate_distance((b.x1, b.y1, b.x2, b.y2), name, cam, priors, method)
                row[method] = round(est.distance_m, 2) if est else None
            lo, hi = next((lo, hi) for lo, hi in BINS if lo <= o["gt_distance"] < hi)
            row["bin"] = f"{lo}-{hi if hi < math.inf else 'inf'} m"
            rows.append(row)

        if i in vis_idx and pairs:
            cv2.imwrite(str(a.out / f"vis_{frame.path.stem}.jpg"), draw(cv2.imread(str(frame.path)), pairs, cam, priors))

    if not rows:
        raise SystemExit("no matched objects to evaluate")
    with open(a.out / "per_object.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    mode = f"detector, {len(rows)}/{n_gt} objects matched" if model else "oracle (projected ground-truth boxes)"
    report = [f"# Distance error on nuScenes {a.version} {a.camera} - {mode}", "",
              "## By class", to_markdown(summarize(rows, "class")), "",
              "## By true distance", to_markdown(summarize(rows, "bin"))]
    (a.out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
