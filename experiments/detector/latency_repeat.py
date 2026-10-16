"""Repeat-measure latency of existing engines, interleaved, to see run-to-run spread.

    python latency_repeat.py --rounds 5

A single latency number per engine hid a large discrepancy: the first INT8 engine
measured 1.52 ms while rebuilt INT8 engines measured about 0.9 ms. This tells
apart a bad build from a noisy measurement by timing every engine several times,
in rotation, within one session on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess

from common import IMGSZ, OUT
from evaluate import median_latency_ms

ENGINES = {
    "PyTorch FP32": OUT / "runs" / "visdrone" / "weights" / "best.pt",
    "FP16 (original build)": OUT / "export" / "fp16" / "best.engine",
    "INT8 MinMax (original build)": OUT / "export" / "int8" / "best.engine",
    "INT8 MinMax (rebuild, calib seed 0)": OUT / "export" / "study_minmax_c0" / "best.engine",
    "INT8 Entropy (calib seed 0)": OUT / "export" / "study_entropy_c0" / "best.engine",
}


ENGINES_B8 = {
    "FP16 batch 8": OUT / "export" / "batch8_fp16" / "best.engine",
    "INT8 MinMax batch 8": OUT / "export" / "batch8_minmax" / "best.engine",
    "INT8 Entropy batch 8": OUT / "export" / "batch8_entropy" / "best.engine",
}


def gpu_state() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--set", choices=["batch1", "batch8"], default="batch1")
    args = ap.parse_args()

    engines, batch = (ENGINES, 1) if args.set == "batch1" else (ENGINES_B8, 8)
    out_name = "latency_repeat.json" if batch == 1 else "latency_repeat_batch8.json"
    samples: dict[str, list[float]] = {k: [] for k in engines}
    states = []
    for rnd in range(args.rounds):
        for name, path in engines.items():
            samples[name].append(median_latency_ms(path, IMGSZ, batch=batch, rounds=1))
        states.append(gpu_state())
        print(f"round {rnd + 1}: " + ", ".join(f"{k.split(' (')[0]}={v[-1]:.2f}" for k, v in samples.items()), flush=True)

    summary = {
        name: {
            "rounds_ms": vals,
            "median_ms": statistics.median(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
        }
        for name, vals in samples.items()
    }
    (OUT / out_name).write_text(
        json.dumps({"gpu_state_after_each_round": states, "engines": summary}, indent=2), encoding="utf-8"
    )
    print()
    for name, s in summary.items():
        print(f"{name:40s} median {s['median_ms']:.2f}  range {s['min_ms']:.2f}-{s['max_ms']:.2f}")
    print("GPU state after each round (sm MHz, mem MHz, W, C, pstate):")
    for s in states:
        print("  ", s)


if __name__ == "__main__":
    main()
