"""Detector accuracy/latency results across precisions, and the comparison table.

Pure Python so it can be tested without a GPU. The GPU experiment in
``experiments/detector`` produces :class:`DetectorResult` records; this module
turns them into the before/after table that goes in the README.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorResult:
    name: str  # e.g. "PyTorch FP32", "TensorRT FP16", "TensorRT INT8"
    map50: float  # 0-1
    map50_95: float  # 0-1
    latency_ms: float  # model inference per image, batch 1
    size_mb: float | None = None  # on-disk model/engine size


@dataclass(frozen=True)
class Comparison:
    name: str
    map50_95_delta: float  # absolute change vs baseline (points on a 0-1 scale)
    map50_95_rel_pct: float  # relative change vs baseline, percent
    speedup: float  # baseline latency / this latency


def compare(result: DetectorResult, baseline: DetectorResult) -> Comparison:
    """How ``result`` differs from ``baseline`` (accuracy cost and speed gain)."""
    if baseline.map50_95 <= 0.0:
        raise ValueError("baseline mAP must be positive to compute a relative change")
    if result.latency_ms <= 0.0 or baseline.latency_ms <= 0.0:
        raise ValueError("latencies must be positive")
    delta = result.map50_95 - baseline.map50_95
    return Comparison(
        name=result.name,
        map50_95_delta=delta,
        map50_95_rel_pct=100.0 * delta / baseline.map50_95,
        speedup=baseline.latency_ms / result.latency_ms,
    )


def to_markdown(results: list[DetectorResult], baseline_index: int = 0) -> str:
    """Table of results, with change and speedup relative to ``results[baseline_index]``."""
    if not results:
        raise ValueError("no results")
    base = results[baseline_index]
    lines = [
        "| Precision | mAP50 | mAP50-95 | Δ mAP50-95 vs. baseline | Latency (ms/img) | Speedup | Size (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        c = compare(r, base)
        is_base = r is base
        delta = "baseline" if is_base else f"{c.map50_95_delta * 100:+.2f} pts ({c.map50_95_rel_pct:+.1f}%)"
        speed = "1.00x" if is_base else f"{c.speedup:.2f}x"
        size = f"{r.size_mb:.1f}" if r.size_mb is not None else "n/a"
        lines.append(
            f"| {r.name} | {r.map50:.3f} | {r.map50_95:.3f} | {delta} | {r.latency_ms:.2f} | {speed} | {size} |"
        )
    return "\n".join(lines)


def save_results(path: str | Path, results: list[DetectorResult], meta: dict | None = None) -> None:
    payload = {"meta": meta or {}, "results": [asdict(r) for r in results]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_results(path: str | Path) -> tuple[list[DetectorResult], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DetectorResult(**r) for r in payload["results"]], payload.get("meta", {})


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (n-1); the std is 0.0 for a single value."""
    if not values:
        raise ValueError("no values")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var**0.5
