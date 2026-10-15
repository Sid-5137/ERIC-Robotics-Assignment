import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from detection.detector import COLORS, Detector
from detection.labels import DISPLAY_NAMES, Box, iou
from distance.estimator import estimate_distance, load_priors
from distance.nuscenes import NuScenes

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class Track:
    _ids = itertools.count(1)

    def __init__(self, box, cls):
        self.id = next(self._ids)
        self.box = np.array(box, dtype=float)
        self.cls = cls
        self.trail = [self.center()]
        self.missed = 0

    def center(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def flow_step(track, prev_gray, gray):
    x1, y1, x2, y2 = track.box.astype(int)
    h, w = gray.shape
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    mask = np.zeros_like(prev_gray)
    mask[y1:y2, x1:x2] = 255
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=30, qualityLevel=0.01, minDistance=2, mask=mask)
    if p0 is None or len(p0) < 3:
        gx, gy = np.meshgrid(np.linspace(x1, x2, 4), np.linspace(y1, y2, 4))
        p0 = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK)
    back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK)
    good = (st.ravel() == 1) & (st_b.ravel() == 1) & (np.linalg.norm(back - p0, axis=2).ravel() < 1.0)
    if good.sum() < 3:
        return False
    a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
    shift = np.median(b - a, axis=0)
    i, j = np.triu_indices(len(a), 1)
    d0, d1 = np.linalg.norm(a[i] - a[j], axis=1), np.linalg.norm(b[i] - b[j], axis=1)
    scale = float(np.median(d1[d0 > 2] / d0[d0 > 2])) if (d0 > 2).any() else 1.0
    cx, cy = track.center()
    hw, hh = (track.box[2] - track.box[0]) / 2 * scale, (track.box[3] - track.box[1]) / 2 * scale
    cx, cy = cx + shift[0], cy + shift[1]
    track.box = np.array([cx - hw, cy - hh, cx + hw, cy + hh])
    track.trail.append((cx, cy))
    return 0 <= cx < w and 0 <= cy < h


def associate(tracks, dets, min_iou=0.3, max_missed=2):
    unmatched = list(dets)
    for t in tracks:
        tb = Box(0, *t.box)
        cands = [d for d in unmatched if d.class_name == t.cls]
        best = max(cands, key=lambda d: iou(tb, Box(0, *d.box)), default=None)
        if best is not None and iou(tb, Box(0, *best.box)) >= min_iou:
            t.box = np.array(best.box, dtype=float)
            t.trail.append(t.center())
            t.missed = 0
            unmatched.remove(best)
        else:
            t.missed += 1
    tracks = [t for t in tracks if t.missed <= max_missed]
    return tracks + [Track(d.box, d.class_name) for d in unmatched]


def draw(frame_bgr, tracks, cam, priors, index, detected):
    out = frame_bgr.copy()
    for t in tracks:
        color = COLORS[t.cls]
        pts = np.array(t.trail[-36:], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        x1, y1, x2, y2 = t.box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        est = estimate_distance(tuple(t.box), t.cls, cam, priors)
        label = f"#{t.id} {DISPLAY_NAMES[t.cls]}" + (f", {est.distance_m:.1f}m" if est else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    status = "detector + flow" if detected else "optical flow only"
    cv2.putText(out, f"frame {index}  {status}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def best_scene(nusc, camera):
    counts = {}
    for i, scene in enumerate(nusc.scenes):
        frames = nusc.sequence(i, camera)
        counts[i] = sum(1 for f in frames if f.is_key_frame for o in nusc.objects(f) if o["class"] == "cone")
    return max(counts, key=counts.get)


def main():
    p = argparse.ArgumentParser(description="Track cones across frames with Lucas-Kanade optical flow.")
    p.add_argument("--root", type=Path, default=Path("data/raw/nuscenes"))
    p.add_argument("--weights", required=True)
    p.add_argument("--scene", type=int, default=None)
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--detect-every", type=int, default=6)
    p.add_argument("--classes", nargs="+", default=["cone"])
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/tracking"))
    a = p.parse_args()

    nusc = NuScenes(a.root)
    scene = a.scene if a.scene is not None else best_scene(nusc, a.camera)
    frames = [f for f in nusc.sequence(scene, a.camera) if f.path.exists()][a.start: a.start + a.frames]
    cam = frames[0].intrinsics()
    detector = Detector(a.weights, cam, conf=0.35, device=a.device)
    priors = load_priors()

    a.out.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames[0].path))
    h, w = first.shape[:2]
    video = cv2.VideoWriter(str(a.out / f"tracking_scene{scene}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    gif_frames, tracks, prev_gray = [], [], None
    for i, f in enumerate(frames):
        img = cv2.imread(str(f.path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            tracks = [t for t in tracks if flow_step(t, prev_gray, gray)]
        detected = i % a.detect_every == 0
        if detected:
            dets = [d for d in detector.detect(img) if d.class_name in a.classes]
            tracks = associate(tracks, dets)
        vis = draw(img, tracks, cam, priors, i, detected)
        video.write(vis)
        if i % 3 == 0:
            gif_frames.append(cv2.cvtColor(cv2.resize(vis, (560, int(560 * h / w))), cv2.COLOR_BGR2RGB))
        prev_gray = gray
    video.release()

    from PIL import Image
    imgs = [Image.fromarray(g) for g in gif_frames]
    imgs[0].save(a.out / f"tracking_scene{scene}.gif", save_all=True, append_images=imgs[1:], duration=250, loop=0, optimize=True)
    print(f"scene {scene}: {len(frames)} frames, {len(tracks)} tracks alive at the end -> {a.out}")


if __name__ == "__main__":
    main()
