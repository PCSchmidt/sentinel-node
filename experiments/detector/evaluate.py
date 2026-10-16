"""Compare PyTorch FP32, TensorRT FP16 and TensorRT INT8 on the same weights.

    python evaluate.py --weights out/runs/visdrone/weights/best.pt
    python evaluate.py --weights out/runs/visdrone_s1/weights/best.pt --tag s1 --int8-entropy

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

from calibration import entropy_calibration
from common import CALIB, IMGSZ, OUT, local_visdrone_yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sentinel_node.detector_report import DetectorResult, save_results, to_markdown  # noqa: E402


def median_latency_ms(model_path: Path, imgsz: int, batch: int = 1, n: int = 300, warmup: int = 30, rounds: int = 3) -> float:
    """Median model-inference time (ms) per image, excluding pre/post-processing.

    Takes the median of ``rounds`` independent medians. A single run proved
    unreliable on a thermally throttling laptop GPU (one reading was 60% off), so
    one number per engine is not trusted. ``batch`` must match the engine's batch.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)
    source = img if batch == 1 else [img] * batch
    round_medians = []
    for _ in range(rounds):
        times = []
        for i in range(warmup + n):
            results = model.predict(source, imgsz=imgsz, device=0, verbose=False)
            if i >= warmup:
                times.append(results[0].speed["inference"])  # per-image time
        round_medians.append(statistics.median(times))
    return statistics.median(round_medians)


def validate(model_path: Path, data: str, imgsz: int) -> tuple[float, float]:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    m = model.val(data=data, imgsz=imgsz, batch=1, device=0, half=False, plots=False, verbose=False)
    return float(m.box.map50), float(m.box.map)


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def export_engine(weights: Path, tag: str, batch: int = 1, **kwargs) -> Path:
    """Export ``weights`` to a TensorRT engine in its own folder; return the engine path."""
    from ultralytics import YOLO

    folder = OUT / "export" / tag
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / "best.pt"
    shutil.copy2(weights, local)
    path = YOLO(str(local)).export(format="engine", imgsz=IMGSZ, batch=batch, device=0, workspace=4, **kwargs)
    return Path(path)


def measure(name: str, path: Path, data: str) -> DetectorResult:
    m50, m5095 = validate(path, data, IMGSZ)
    return DetectorResult(name, m50, m5095, median_latency_ms(path, IMGSZ), size_mb(path))


def environment_meta(weights: Path, extra: dict | None = None) -> dict:
    import tensorrt
    import torch
    import ultralytics

    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tensorrt": tensorrt.__version__,
        "ultralytics": ultralytics.__version__,
        "imgsz": IMGSZ,
        "weights": str(weights),
        "latency": "median of 3 rounds, each the median of 300 single-image predictions after 30 warm-up runs, model inference only, batch 1",
    }
    meta.update(extra or {})
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--calib-data", type=Path, default=CALIB / "calib.yaml")
    ap.add_argument("--tag", default="", help="suffix for output files and export folders")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--int8-entropy", action="store_true", help="also build an INT8 engine with entropy calibration")
    args = ap.parse_args()

    data = local_visdrone_yaml()
    sfx = f"_{args.tag}" if args.tag else ""
    if not args.calib_data.exists():
        raise SystemExit("run make_calib.py first (held-out calibration set)")

    results: list[DetectorResult] = []

    print("== PyTorch FP32 ==")
    m50, m5095 = validate(args.weights, data, IMGSZ)
    results.append(DetectorResult("PyTorch FP32", m50, m5095, median_latency_ms(args.weights, IMGSZ), size_mb(args.weights)))

    print("== TensorRT FP16 ==")
    fp16 = export_engine(args.weights, f"fp16{sfx}", half=True)
    results.append(measure("TensorRT FP16", fp16, data))

    if not args.skip_int8:
        print("== TensorRT INT8 (MinMax) ==")
        int8 = export_engine(args.weights, f"int8{sfx}", int8=True, data=str(args.calib_data))
        results.append(measure("TensorRT INT8 (MinMax)" if args.int8_entropy else "TensorRT INT8", int8, data))

    if args.int8_entropy:
        print("== TensorRT INT8 (Entropy) ==")
        with entropy_calibration():
            ent = export_engine(args.weights, f"int8_entropy{sfx}", int8=True, data=str(args.calib_data))
        results.append(measure("TensorRT INT8 (Entropy)", ent, data))

    meta = environment_meta(
        args.weights,
        {
            "dataset": data,
            "int8_calibration": "Ultralytics EngineCalibrator; images from calib.yaml (held-out training sample); "
            "MinMax unless labelled Entropy",
        },
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save_results(OUT / f"results{sfx}.json", results, meta)
    table = to_markdown(results)
    (OUT / f"results{sfx}.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
