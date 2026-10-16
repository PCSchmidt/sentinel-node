"""Compare PyTorch FP32, TensorRT FP16 and TensorRT INT8 on the same weights.

    python evaluate.py --weights out/runs/visdrone/weights/best.pt

For each precision: mAP50 and mAP50-95 on the VisDrone validation split, median
single-image inference latency, and model size. Engines are built from the same
checkpoint; INT8 is calibrated on held-out training images (see make_calib.py).

Latency here is measured on whatever GPU this runs on. It is NOT a Jetson
number, and TensorRT engines are not portable: the Jetson build is redone there.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

from common import CALIB, IMGSZ, OUT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sentinel_node.detector_report import DetectorResult, save_results, to_markdown  # noqa: E402


def median_latency_ms(model_path: Path, imgsz: int, n: int = 300, warmup: int = 30) -> float:
    """Median model-inference time (ms) for one image, excluding pre/post-processing."""
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)
    times = []
    for i in range(warmup + n):
        r = model.predict(img, imgsz=imgsz, device=0, verbose=False)[0]
        if i >= warmup:
            times.append(r.speed["inference"])
    return statistics.median(times)


def validate(model_path: Path, data: str, imgsz: int) -> tuple[float, float]:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    m = model.val(data=data, imgsz=imgsz, batch=1, device=0, half=False, plots=False, verbose=False)
    return float(m.box.map50), float(m.box.map)


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def export_engine(weights: Path, tag: str, **kwargs) -> Path:
    """Export ``weights`` to a TensorRT engine in its own folder; return the engine path."""
    from ultralytics import YOLO

    folder = OUT / "export" / tag
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / "best.pt"
    shutil.copy2(weights, local)
    path = YOLO(str(local)).export(format="engine", imgsz=IMGSZ, batch=1, device=0, workspace=4, **kwargs)
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--calib-data", type=Path, default=CALIB / "calib.yaml")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    import torch
    import ultralytics

    from common import local_visdrone_yaml

    DATA = local_visdrone_yaml()
    if not args.calib_data.exists():
        raise SystemExit("run make_calib.py first (held-out calibration set)")

    results: list[DetectorResult] = []

    print("== PyTorch FP32 ==")
    m50, m5095 = validate(args.weights, DATA, IMGSZ)
    results.append(DetectorResult("PyTorch FP32", m50, m5095, median_latency_ms(args.weights, IMGSZ), size_mb(args.weights)))

    print("== TensorRT FP16 ==")
    fp16 = export_engine(args.weights, "fp16", half=True)
    m50, m5095 = validate(fp16, DATA, IMGSZ)
    results.append(DetectorResult("TensorRT FP16", m50, m5095, median_latency_ms(fp16, IMGSZ), size_mb(fp16)))

    if not args.skip_int8:
        print("== TensorRT INT8 ==")
        int8 = export_engine(args.weights, "int8", int8=True, data=str(args.calib_data))
        m50, m5095 = validate(int8, DATA, IMGSZ)
        results.append(DetectorResult("TensorRT INT8", m50, m5095, median_latency_ms(int8, IMGSZ), size_mb(int8)))

    import tensorrt

    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tensorrt": tensorrt.__version__,
        "ultralytics": ultralytics.__version__,
        "dataset": DATA,
        "imgsz": IMGSZ,
        "weights": str(args.weights),
        "int8_calibration": "Ultralytics EngineCalibrator, TensorRT MinMax; images from calib.yaml (held-out training sample)",
        "latency": "median of 300 single-image predictions after 30 warm-up runs, model inference only, batch 1",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    save_results(OUT / "results.json", results, meta)
    table = to_markdown(results)
    (OUT / "results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
