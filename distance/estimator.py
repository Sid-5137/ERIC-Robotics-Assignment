import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from detection.labels import CLASS_NAMES
from distance.camera import CameraIntrinsics

EDGE_PX = 2


@dataclass(frozen=True)
class SizePrior:
    height_m: float
    use: str = "height"


def load_priors(path="configs/objects.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    return {name: SizePrior(**cfg[name]) for name in CLASS_NAMES}


@dataclass(frozen=True)
class DistanceEstimate:
    distance_m: float
    depth_m: float
    method: str
    reliable: bool = True


def _range_from_depth(depth, u, v, cam: CameraIntrinsics) -> float:
    x = (u - cam.cx) / cam.fx
    y = (v - cam.cy) / cam.fy
    return depth * math.sqrt(1.0 + x * x + y * y)


def size_method(box, prior: SizePrior, cam: CameraIntrinsics) -> Optional[DistanceEstimate]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if prior.use == "max" and w > h:
        pixels, f = w, cam.fx
    else:
        pixels, f = h, cam.fy
    if pixels <= 1:
        return None
    depth = f * prior.height_m / pixels
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    cut_off = y1 <= EDGE_PX or y2 >= cam.height - EDGE_PX
    if prior.use == "max":
        cut_off = cut_off or x1 <= EDGE_PX or x2 >= cam.width - EDGE_PX
    return DistanceEstimate(_range_from_depth(depth, u, v, cam), depth, "size", reliable=not cut_off)


def ground_method(box, prior: SizePrior, cam: CameraIntrinsics) -> Optional[DistanceEstimate]:
    if cam.camera_height_m is None:
        return None
    x1, _, x2, y2 = box
    if y2 >= cam.height - EDGE_PX:
        return None
    p = math.radians(cam.pitch_deg)
    rx, ry, rz = ((x1 + x2) / 2 - cam.cx) / cam.fx, (y2 - cam.cy) / cam.fy, 1.0
    ly = ry * math.cos(p) + rz * math.sin(p)
    lz = -ry * math.sin(p) + rz * math.cos(p)
    if ly <= 1e-3:
        return None
    t = cam.camera_height_m / ly
    gx, gz = rx * t, lz * t
    cy_obj = cam.camera_height_m - prior.height_m / 2
    distance = math.sqrt(gx * gx + cy_obj * cy_obj + gz * gz)
    depth = gz * math.cos(p) + cy_obj * math.sin(p)
    return DistanceEstimate(distance, depth, "ground")


def estimate_distance(box, class_name: str, cam: CameraIntrinsics, priors: dict,
                      method: str = "size") -> Optional[DistanceEstimate]:
    prior = priors[class_name]
    size = size_method(box, prior, cam)
    if method == "size":
        return size
    ground = ground_method(box, prior, cam)
    if method == "ground":
        return ground
    if size is None or not size.reliable:
        return ground or size
    if ground is None or not (1 / 1.5 <= ground.distance_m / size.distance_m <= 1.5):
        return size
    return DistanceEstimate((size.distance_m + ground.distance_m) / 2,
                            (size.depth_m + ground.depth_m) / 2, "fused")
