import argparse
import json
from pathlib import Path

import cv2

from detection.detector import Detector, annotate
from distance.camera import CameraIntrinsics

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="image, folder, video file or webcam index")
    p.add_argument("--camera", default="configs/camera_bdd100k.yaml", help="default intrinsics YAML")
    p.add_argument("--intrinsics", type=Path, help="JSON {image_stem: 3x3 K} for per-image calibration")
    p.add_argument("--method", choices=["size", "ground", "fused"], default="size")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/infer"))
    p.add_argument("--max-frames", type=int, default=0)
    a = p.parse_args()

    default_cam = CameraIntrinsics.from_yaml(a.camera)
    per_image = json.loads(a.intrinsics.read_text()) if a.intrinsics else {}
    det = Detector(a.weights, default_cam, conf=a.conf, imgsz=a.imgsz, device=a.device, method=a.method)
    a.out.mkdir(parents=True, exist_ok=True)

    src = Path(a.source)
    is_video = src.suffix.lower() in VIDEO_EXTS or a.source.isdigit()
    writer, records = None, []

    for i, (frame, dets, path) in enumerate(det.stream(a.source)):
        stem = Path(path).stem
        if stem in per_image:
            h, w = frame.shape[:2]
            dets = det.redistance(dets, CameraIntrinsics.from_matrix(per_image[stem], w, h))
        vis = annotate(frame, dets)
        if is_video:
            if writer is None:
                h, w = vis.shape[:2]
                name = "webcam" if a.source.isdigit() else src.stem
                writer = cv2.VideoWriter(str(a.out / f"{name}_annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(vis)
        else:
            cv2.imwrite(str(a.out / f"{stem}.jpg"), vis)
        records.append({"source": str(path), "frame": i, "detections": [d.to_dict() for d in dets]})
        if a.max_frames and i + 1 >= a.max_frames:
            break

    if writer:
        writer.release()
    (a.out / "detections.json").write_text(json.dumps(records, indent=1))
    print(f"saved {len(records)} annotated frame(s) + detections.json to {a.out}")


if __name__ == "__main__":
    main()
