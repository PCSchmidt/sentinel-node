"""Milestone 0.3b studies on one trained model (seed 0).

    python calib_study.py --weights out/runs/visdrone/weights/best.pt --stage calib
    python calib_study.py --weights out/runs/visdrone/weights/best.pt --stage batch

``calib``: INT8 accuracy for MinMax vs. Entropy calibration, each with three
different random 512-image calibration samples (seeds 0, 1, 2), to separate the
effect of the calibration algorithm from the luck of the sample.

``batch``: per-image latency of FP16 and INT8 engines at batch size 8, to test
whether INT8's lack of speed-up at batch 1 comes from a launch-bound small model.
Run on an otherwise idle GPU; latency readings are disturbed by other GPU work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration import entropy_calibration
from common import CALIB, IMGSZ, OUT, local_visdrone_yaml
from evaluate import environment_meta, export_engine, measure, median_latency_ms
from make_calib import make_calib

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sentinel_node.detector_report import DetectorResult, save_results  # noqa: E402

CALIB_SEEDS = (0, 1, 2)


def calib_yaml_for(seed: int) -> Path:
    if seed == 0:
        return CALIB / "calib.yaml"  # the sample used for the main result
    return make_calib(512, seed, CALIB / f"seed{seed}")


def stage_calib(weights: Path) -> None:
    data = local_visdrone_yaml()
    results: list[DetectorResult] = []
    for algo in ("minmax", "entropy"):
        for seed in CALIB_SEEDS:
            calib = calib_yaml_for(seed)
            tag = f"study_{algo}_c{seed}"
            print(f"== INT8 {algo}, calibration seed {seed} ==", flush=True)
            if algo == "entropy":
                with entropy_calibration():
                    engine = export_engine(weights, tag, int8=True, data=str(calib))
            else:
                engine = export_engine(weights, tag, int8=True, data=str(calib))
            results.append(measure(f"INT8 {algo} calib-seed {seed}", engine, data))
            save_results(OUT / "study_calibration.json", results, environment_meta(weights, {"dataset": data}))


def stage_batch(weights: Path, batch: int = 8) -> None:
    calib = str(CALIB / "calib.yaml")
    rows = []
    for name, kwargs, entropy in (
        ("TensorRT FP16", {"half": True}, False),
        ("TensorRT INT8 (MinMax)", {"int8": True, "data": calib}, False),
        ("TensorRT INT8 (Entropy)", {"int8": True, "data": calib}, True),
    ):
        print(f"== {name}, batch {batch} ==", flush=True)
        tag = f"batch{batch}_{name.split()[-1].strip('()').lower()}"
        if entropy:
            with entropy_calibration():
                engine = export_engine(weights, tag, batch=batch, **kwargs)
        else:
            engine = export_engine(weights, tag, batch=batch, **kwargs)
        rows.append({"name": name, "batch": batch, "latency_ms_per_image": median_latency_ms(engine, IMGSZ, batch=batch)})
        (OUT / "study_batch.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--stage", choices=["calib", "batch"], required=True)
    args = ap.parse_args()
    if args.stage == "calib":
        stage_calib(args.weights)
    else:
        stage_batch(args.weights)


if __name__ == "__main__":
    main()
