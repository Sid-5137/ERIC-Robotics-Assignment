import argparse
from pathlib import Path

import cv2
import numpy as np

from detection.detector import COLORS, Detector, annotate
from distance.nuscenes import NuScenes

LEGEND = "filled: size-based distance   ring: ground-plane (homography)   x: LiDAR ground truth"


class BirdsEye:
    def __init__(self, frame, max_range=40.0, half_width=15.0, px_per_m=15):
        self.frame, self.R, self.s = frame, max_range, px_per_m
        self.size = (int(2 * half_width * px_per_m), int(max_range * px_per_m))
        w = self.size[0]
        self.A = np.array([[0, -1 / px_per_m, max_range],
                           [-1 / px_per_m, 0, w / (2 * px_per_m)],
                           [0, 0, 1]])
        self.H = frame.ground_homography()

    def warp(self, image):
        return cv2.warpPerspective(image, self.H @ self.A, self.size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def to_px(self, x, y):
        return int(round(self.size[0] / 2 - y * self.s)), int(round((self.R - x) * self.s))

    def ground_point(self, u, v):
        p = np.linalg.inv(self.H) @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def along_ray(self, u, v, distance):
        f = self.frame
        ray = np.linalg.inv(f.K) @ np.array([u, v, 1.0])
        p = f.R_cs @ (ray / np.linalg.norm(ray) * distance) + f.t_cs
        return p[0], p[1]

    def grid(self, bev):
        ego = self.to_px(0, 0)
        for r in range(10, int(self.R) + 1, 10):
            cv2.circle(bev, ego, r * self.s, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(bev, f"{r} m", (ego[0] + 4, ego[1] - r * self.s + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(bev, (ego[0], ego[1] - 4), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
        return bev


def render(nusc, frame, detector, show_gt=True):
    img = cv2.imread(str(frame.path))
    dets = detector.detect(img, camera=frame.intrinsics())
    view = BirdsEye(frame)
    bev = view.grid((view.warp(img) * 0.6).astype(np.uint8))

    if show_gt:
        for _, a in nusc.annotations.get(frame.sample_token, []):
            x, y, _ = frame.R_ego.T @ (np.array(a["translation"]) - frame.t_ego)
            if 0 < x < view.R:
                cv2.drawMarker(bev, view.to_px(x, y), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)

    for d in dets:
        if d.distance is None:
            continue
        x1, y1, x2, y2 = d.box
        color = COLORS[d.class_name]
        cv2.circle(bev, view.to_px(*view.along_ray((x1 + x2) / 2, (y1 + y2) / 2, d.distance.distance_m)),
                   6, color, -1, cv2.LINE_AA)
        if y2 < frame.height - 2:
            cv2.circle(bev, view.to_px(*view.ground_point((x1 + x2) / 2, y2)), 6, color, 1, cv2.LINE_AA)

    cam_img = annotate(img, dets)
    cam_img = cv2.resize(cam_img, (int(cam_img.shape[1] * bev.shape[0] / cam_img.shape[0]), bev.shape[0]))
    out = np.hstack([cam_img, bev])
    cv2.putText(out, LEGEND, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out, dets


def best_frames(nusc, n):
    best = {}
    for f in nusc.key_frames():
        scene = nusc.samples[f.sample_token]["scene_token"]
        count = len(nusc.objects(f))
        if count and count > best.get(scene, (0, None))[0]:
            best[scene] = (count, f)
    return [f for _, f in sorted(best.values(), key=lambda x: -x[0])][:n]


def images(nusc, detector, n, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in best_frames(nusc, n):
        vis, dets = render(nusc, f, detector)
        cv2.imwrite(str(out / f"bev_{f.path.stem}.jpg"), vis)
        print(f"{f.path.name}: {len(dets)} detections")


def video(nusc, detector, scene_index, start, count, out, fps=12, width=1600, crf=28, gif_seconds=6, gif_width=640):
    import imageio_ffmpeg
    from PIL import Image

    scene = nusc.scenes[scene_index]
    frames = [f for f in nusc.sequence(scene_index) if f.path.exists()][start:start + count]
    writer, gif, t0 = None, [], frames[0].timestamp
    for i, f in enumerate(frames):
        vis, dets = render(nusc, f, detector, show_gt=f.is_key_frame)
        vis = cv2.resize(vis, (width // 16 * 16, int(vis.shape[0] * width / vis.shape[1]) // 16 * 16))
        title = f"{scene['name']}  t={(f.timestamp - t0) / 1e6:5.1f}s  {len(dets)} detections"
        if f.is_key_frame:
            title += "   LiDAR key frame"
        bar = np.zeros((48, vis.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(np.vstack([bar, vis]), cv2.COLOR_BGR2RGB)
        if writer is None:
            writer = imageio_ffmpeg.write_frames(str(out), (rgb.shape[1], rgb.shape[0]), fps=fps, codec="libx264",
                                                 pix_fmt_out="yuv420p", output_params=["-crf", str(crf), "-preset", "slow"])
            writer.send(None)
        writer.send(np.ascontiguousarray(rgb))
        if i < gif_seconds * fps and i % 2 == 0:
            small = cv2.resize(rgb, (gif_width, int(rgb.shape[0] * gif_width / rgb.shape[1])))
            gif.append(Image.fromarray(small).quantize(128))
    writer.close()
    gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=2000 // fps, loop=0, optimize=True)
    print(f"{scene['name']}: {len(frames)} frames -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Top-down view of detections through the ground homography, as images or video.")
    p.add_argument("--weights", required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/nuscenes"))
    p.add_argument("--device", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    im = sub.add_parser("images", help="one frame per scene, the one with the most cones/barriers")
    im.add_argument("--n", type=int, default=6)
    im.add_argument("--out", type=Path, default=Path("outputs/bev"))
    vd = sub.add_parser("video", help="side-by-side H.264 video of a whole scene plus a GIF preview")
    vd.add_argument("--scene", type=int, required=True)
    vd.add_argument("--start", type=int, default=0)
    vd.add_argument("--frames", type=int, default=240)
    vd.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    nusc = NuScenes(a.root)
    first = next(nusc.key_frames())
    detector = Detector(a.weights, first.intrinsics(), conf=0.3, device=a.device)
    if a.cmd == "images":
        images(nusc, detector, a.n, a.out)
    else:
        video(nusc, detector, a.scene, a.start, a.frames, a.out)
