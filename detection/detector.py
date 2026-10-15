from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from detection.labels import DISPLAY_NAMES
from distance.camera import CameraIntrinsics
from distance.estimator import DistanceEstimate, estimate_distance, load_priors

COLORS = {"cone": (0, 140, 255), "barrier": (0, 215, 255), "stop_sign": (40, 40, 230)}


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple
    distance: Optional[DistanceEstimate]

    @property
    def label(self) -> str:
        name = DISPLAY_NAMES[self.class_name]
        if self.distance is None:
            return name
        approx = "" if self.distance.reliable else "~"
        return f"{name}, {approx}{self.distance.distance_m:.1f}m"

    def to_dict(self) -> dict:
        d = {"class": self.class_name, "confidence": round(self.confidence, 3),
             "box": [round(v, 1) for v in self.box]}
        if self.distance:
            d.update(distance_m=round(self.distance.distance_m, 2), depth_m=round(self.distance.depth_m, 2),
                     method=self.distance.method, reliable=self.distance.reliable)
        return d


class Detector:
    def __init__(self, weights: str, camera: CameraIntrinsics, priors_path="configs/objects.yaml",
                 conf=0.35, imgsz=640, device=None, method="size"):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.camera = camera
        self.priors = load_priors(priors_path)
        self.conf, self.imgsz, self.device, self.method = conf, imgsz, device, method

    def _to_detections(self, result, camera: Optional[CameraIntrinsics]) -> list:
        h, w = result.orig_shape
        cam = (camera or self.camera).scaled_to(w, h)
        names = result.names
        dets = []
        for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist()):
            name = names[int(cls)]
            dist = estimate_distance(xyxy, name, cam, self.priors, self.method)
            dets.append(Detection(name, conf, tuple(xyxy), dist))
        return dets

    def redistance(self, detections: list, camera: CameraIntrinsics) -> list:
        return [Detection(d.class_name, d.confidence, d.box,
                          estimate_distance(d.box, d.class_name, camera, self.priors, self.method))
                for d in detections]

    def detect(self, image, camera: Optional[CameraIntrinsics] = None) -> list:
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return self._to_detections(result, camera)

    def stream(self, source):
        for r in self.model.predict(source, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                    stream=True, verbose=False):
            yield r.orig_img, self._to_detections(r, None), r.path


def annotate(image: np.ndarray, detections: list) -> np.ndarray:
    out = image.copy()
    scale = max(0.5, min(out.shape[:2]) / 900)
    thick = max(1, int(round(2 * scale)))
    font_scale = 0.55 * scale
    placed = []
    for d in sorted(detections, key=lambda d: -(d.box[3] - d.box[1])):
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(d.class_name, (0, 255, 0)), thick)
    for d in sorted(detections, key=lambda d: -(d.box[3] - d.box[1])):
        x1, y1, x2, y2 = map(int, d.box)
        color = COLORS.get(d.class_name, (0, 255, 0))
        (tw, th), base = cv2.getTextSize(d.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        h = th + base + 4
        ty = y1 - 2 if y1 - h > 0 else y2 + h
        tx = min(x1, out.shape[1] - tw - 4)
        while any(tx < px2 and tx + tw + 4 > px1 and ty - h < py2 and ty > py1 for px1, py1, px2, py2 in placed):
            ty -= h
        if ty - h < 0:
            continue
        placed.append((tx, ty - h, tx + tw + 4, ty))
        cv2.rectangle(out, (tx, ty - h), (tx + tw + 4, ty), color, -1)
        cv2.line(out, (tx, ty), (x1, y1), color, 1)
        cv2.putText(out, d.label, (tx + 2, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thick, cv2.LINE_AA)
    return out
