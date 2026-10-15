import argparse
import glob
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    camera_height_m: Optional[float] = None
    pitch_deg: float = 0.0

    @classmethod
    def from_fov(cls, width, height, hfov_deg, **kw):
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(f, f, width / 2, height / 2, width, height, **kw)

    @classmethod
    def from_matrix(cls, K, width, height, **kw):
        return cls(K[0][0], K[1][1], K[0][2], K[1][2], width, height, **kw)

    @classmethod
    def from_yaml(cls, path):
        cfg = yaml.safe_load(Path(path).read_text())
        kw = {"camera_height_m": cfg.get("camera_height_m"), "pitch_deg": cfg.get("pitch_deg", 0.0)}
        w, h = cfg["image_width"], cfg["image_height"]
        if "fx" in cfg:
            return cls(cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"], w, h, **kw)
        return cls.from_fov(w, h, cfg["hfov_deg"], **kw)

    def scaled_to(self, width, height):
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
                       width=width, height=height)


def calibrate(images, pattern=(9, 6), square_m=0.025):
    import cv2
    import numpy as np

    cols, rows = pattern
    board = np.zeros((cols * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    obj_pts, img_pts, size = [], [], None
    for f in images:
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if found:
            obj_pts.append(board)
            img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
    print(f"checkerboard found in {len(obj_pts)}/{len(images)} images")
    if len(obj_pts) < 8:
        raise SystemExit("need at least ~8 good views for a stable calibration")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"RMS reprojection error {rms:.3f} px (below 0.5 is good)")
    return K, dist.ravel(), rms, size


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One-time checkerboard calibration of the robot camera.")
    p.add_argument("--images", nargs="+", required=True, help="e.g. calib/*.jpg")
    p.add_argument("--pattern", type=int, nargs=2, default=[9, 6], help="inner corners per row and column")
    p.add_argument("--square", type=float, default=0.025, help="square size in metres")
    p.add_argument("--camera-height", type=float, help="lens height above the floor in metres")
    p.add_argument("--pitch", type=float, default=0.0, help="downward tilt in degrees")
    p.add_argument("--out", default="configs/camera_robot.yaml")
    a = p.parse_args()

    files = [f for pattern in a.images for f in glob.glob(pattern)]
    K, dist, rms, (w, h) = calibrate(files, tuple(a.pattern), a.square)
    cfg = {"image_width": w, "image_height": h, "fx": float(K[0, 0]), "fy": float(K[1, 1]),
           "cx": float(K[0, 2]), "cy": float(K[1, 2]), "distortion": [float(d) for d in dist],
           "rms_px": float(rms), "camera_height_m": a.camera_height, "pitch_deg": a.pitch}
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"saved {a.out}")
