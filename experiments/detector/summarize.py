"""Build the milestone 0.3b tables from the result files in ``out/``.

    python summarize.py

Reads whichever of these exist: results.json (seed 0, main), results_s1.json,
results_s2.json (other training seeds), study_calibration.json, study_batch.json.
Writes ``out/summary_0_3b.md`` and prints it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import OUT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sentinel_node.detector_report import DetectorResult, load_results, mean_std  # noqa: E402


def pts(x: float) -> str:
    return f"{x * 100:+.2f}"


def by_name(results: list[DetectorResult]) -> dict[str, DetectorResult]:
    # Seed 0 was run before entropy calibration existed, so its MinMax row is
    # just "TensorRT INT8"; later runs label it explicitly. Treat them as one.
    return {r.name.replace(" (MinMax)", ""): r for r in results}


def calibration_table(base: DetectorResult) -> str | None:
    path = OUT / "study_calibration.json"
    if not path.exists():
        return None
    results, _ = load_results(path)
    lines = [
        "| INT8 calibration | Calibration sample (seed) | mAP50-95 | Δ vs. FP32 (pts) |",
        "| --- | --- | --- | --- |",
    ]
    groups: dict[str, list[float]] = {}
    for r in results:
        _, algo, _, seed = r.name.split()
        lines.append(f"| {algo} | {seed} | {r.map50_95:.4f} | {pts(r.map50_95 - base.map50_95)} |")
        groups.setdefault(algo, []).append(r.map50_95 - base.map50_95)
    lines += ["", "| INT8 calibration | n samples | Mean Δ vs. FP32 (pts) | Std (pts) | Range (pts) |", "| --- | --- | --- | --- | --- |"]
    for algo, deltas in groups.items():
        m, s = mean_std(deltas)
        lines.append(f"| {algo} | {len(deltas)} | {pts(m)} | {s * 100:.2f} | {pts(min(deltas))} to {pts(max(deltas))} |")
    return "\n".join(lines)


def seeds_table() -> str | None:
    runs = []
    for tag, file in (("0", "results.json"), ("1", "results_s1.json"), ("2", "results_s2.json")):
        path = OUT / file
        if path.exists():
            runs.append((tag, by_name(load_results(path)[0])))
    if len(runs) < 2:
        return None
    precisions = [n for n in runs[0][1] if n != "PyTorch FP32"] + []
    for _, d in runs[1:]:
        precisions += [n for n in d if n not in precisions and n != "PyTorch FP32"]
    lines = ["| Training seed | " + " | ".join(["FP32 mAP50-95"] + [f"{p} Δ (pts)" for p in precisions]) + " |"]
    lines.append("| --- | " + " | ".join(["---"] * (1 + len(precisions))) + " |")
    deltas: dict[str, list[float]] = {p: [] for p in precisions}
    fp32s = []
    for tag, d in runs:
        base = d["PyTorch FP32"]
        fp32s.append(base.map50_95)
        cells = [f"{base.map50_95:.4f}"]
        for p in precisions:
            if p in d:
                delta = d[p].map50_95 - base.map50_95
                deltas[p].append(delta)
                cells.append(pts(delta))
            else:
                cells.append("n/a")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    m, s = mean_std(fp32s)
    cells = [f"{m:.4f} ± {s:.4f}"]
    for p in precisions:
        if deltas[p]:
            pm, ps = mean_std(deltas[p])
            cells.append(f"{pts(pm)} ± {ps * 100:.2f} (n={len(deltas[p])})")
        else:
            cells.append("n/a")
    lines.append("| **mean ± std** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latency_table() -> str | None:
    """Same-round latency ratios vs. FP16 from the interleaved repeat measurements.

    Absolute latency drifts together across engines as the laptop GPU heats and
    throttles, so engines are compared within each round and the median ratio is
    reported, with the range of absolute values for context.
    """
    import statistics

    lines = [
        "| Engine | Batch | Median ms/img | Range (ms) | Median same-round ratio vs. FP16 |",
        "| --- | --- | --- | --- | --- |",
    ]
    found = False
    for file, batch, base_key in (
        ("latency_repeat.json", 1, "FP16 (original build)"),
        ("latency_repeat_batch8.json", 8, "FP16 batch 8"),
    ):
        path = OUT / file
        if not path.exists():
            continue
        found = True
        engines = json.loads(path.read_text(encoding="utf-8"))["engines"]
        base = engines[base_key]["rounds_ms"]
        for name, d in engines.items():
            ratio = statistics.median(v / b for v, b in zip(d["rounds_ms"], base))
            lines.append(
                f"| {name} | {batch} | {d['median_ms']:.2f} | {d['min_ms']:.2f}-{d['max_ms']:.2f} | {ratio:.2f} |"
            )
    return "\n".join(lines) if found else None


def main() -> None:
    main_results, _ = load_results(OUT / "results.json")
    base = by_name(main_results)["PyTorch FP32"]
    parts = []
    for title, table in (
        ("INT8 calibration algorithm and sample (seed-0 model)", calibration_table(base)),
        ("Across training seeds", seeds_table()),
        ("Latency, interleaved rounds (seed-0 model)", latency_table()),
    ):
        if table:
            parts.append(f"### {title}\n\n{table}\n")
    text = "\n".join(parts)
    (OUT / "summary_0_3b.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
