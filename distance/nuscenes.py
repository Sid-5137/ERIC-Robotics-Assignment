import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from detection.labels import Box
from distance.camera import CameraIntrinsics

CATEGORY_MAP = {"movable_object.trafficcone": "cone", "movable_object.barrier": "barrier"}


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_corners(ann):
    w, l, h = ann["size"]
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return quat_to_rot(ann["rotation"]) @ np.vstack([x, y, z]) + np.array(ann["translation"])[:, None]


@dataclass
class CameraFrame:
    token: str
    path: Path
    sample_token: str
    timestamp: int
    is_key_frame: bool
    width: int
    height: int
    K: np.ndarray
    R_cs: np.ndarray
    t_cs: np.ndarray
    R_ego: np.ndarray
    t_ego: np.ndarray

    def global_to_cam(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(3, -1)
        return self.R_cs.T @ (self.R_ego.T @ (pts - self.t_ego[:, None]) - self.t_cs[:, None])

    def intrinsics(self) -> CameraIntrinsics:
        axis = self.R_cs @ np.array([0.0, 0.0, 1.0])
        return CameraIntrinsics.from_matrix(self.K, self.width, self.height,
                                            camera_height_m=float(self.t_cs[2]),
                                            pitch_deg=math.degrees(math.asin(-axis[2])))

    def ground_homography(self):
        R_ec, t_ec = self.R_cs.T, -self.R_cs.T @ self.t_cs
        return self.K @ np.column_stack([R_ec[:, 0], R_ec[:, 1], t_ec])


class NuScenes:
    def __init__(self, root, version="v1.0-mini"):
        self.root = Path(root)
        self.version = version

    def _table(self, name):
        return json.loads((self.root / self.version / f"{name}.json").read_text())

    @cached_property
    def sample_data(self):
        return {sd["token"]: sd for sd in self._table("sample_data")}

    @cached_property
    def calib(self):
        return {c["token"]: c for c in self._table("calibrated_sensor")}

    @cached_property
    def poses(self):
        return {p["token"]: p for p in self._table("ego_pose")}

    @cached_property
    def samples(self):
        return {s["token"]: s for s in self._table("sample")}

    @cached_property
    def scenes(self):
        return self._table("scene")

    @cached_property
    def annotations(self):
        cats = {c["token"]: c["name"] for c in self._table("category")}
        inst_cat = {i["token"]: cats[i["category_token"]] for i in self._table("instance")}
        by_sample = defaultdict(list)
        for a in self._table("sample_annotation"):
            name = CATEGORY_MAP.get(inst_cat[a["instance_token"]])
            if name:
                by_sample[a["sample_token"]].append((name, a))
        return by_sample

    def frame(self, token) -> CameraFrame:
        sd = self.sample_data[token]
        cs, pose = self.calib[sd["calibrated_sensor_token"]], self.poses[sd["ego_pose_token"]]
        return CameraFrame(token, self.root / sd["filename"], sd["sample_token"], sd["timestamp"],
                           sd["is_key_frame"], sd["width"], sd["height"], np.array(cs["camera_intrinsic"]),
                           quat_to_rot(cs["rotation"]), np.array(cs["translation"]),
                           quat_to_rot(pose["rotation"]), np.array(pose["translation"]))

    def key_frames(self, camera="CAM_FRONT"):
        for token, sd in self.sample_data.items():
            if sd["is_key_frame"] and f"/{camera}/" in sd["filename"]:
                yield self.frame(token)

    def sequence(self, scene_index, camera="CAM_FRONT"):
        scene = self.scenes[scene_index]
        first = self.samples[scene["first_sample_token"]]
        start = next(t for t, sd in self.sample_data.items()
                     if sd["sample_token"] == first["token"] and f"/{camera}/" in sd["filename"])
        token, frames = start, []
        while token:
            frames.append(self.frame(token))
            token = self.sample_data[token]["next"]
        return frames

    def objects(self, frame: CameraFrame, min_visibility=3):
        out = []
        for name, a in self.annotations.get(frame.sample_token, []):
            if int(a["visibility_token"] or 0) < min_visibility or a["num_lidar_pts"] == 0:
                continue
            c = frame.global_to_cam(box_corners(a))
            if (c[2] < 0.5).any():
                continue
            uv = (frame.K @ c)[:2] / c[2]
            box = Box(0, uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max())
            clipped = box.clip(frame.width, frame.height)
            if clipped.w * clipped.h < 0.6 * box.w * box.h or clipped.h < 8:
                continue
            centre = frame.global_to_cam(np.array(a["translation"]))
            out.append({"class": name, "box": clipped, "gt_distance": float(np.linalg.norm(centre)),
                        "gt_height": a["size"][2]})
        return out
