"""Per-stage latency and energy measurement.

``StageTimer`` records wall-clock time for named pipeline stages and
summarises them as percentiles. Energy is derived as average power times
latency; the power reading comes from a ``PowerSource``. On a Jetson that would
read board power while the pipeline runs. No power source is implemented here
yet, so energy columns stay empty until one is attached on device.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted, non-empty list."""
    if not sorted_values:
        raise ValueError("no samples")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


@dataclass(frozen=True)
class StageSummary:
    stage: str
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class PowerSource(Protocol):
    """Anything that can report instantaneous board power in watts."""

    def watts(self) -> float | None: ...


_TEGRASTATS_VDD_IN = re.compile(r"VDD_IN\s+(\d+)\s*mW")


def parse_tegrastats_power(line: str) -> float | None:
    """Total input power in watts from one ``tegrastats`` line, if present.

    The ``VDD_IN <n>mW`` field name is taken from Jetson documentation and
    still has to be checked against real output on the target device.
    """
    m = _TEGRASTATS_VDD_IN.search(line)
    return int(m.group(1)) / 1000.0 if m else None


class StageTimer:
    """Collects per-stage durations across frames."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self._samples[name].append((self._clock() - start) * 1000.0)

    def record(self, name: str, ms: float) -> None:
        self._samples[name].append(ms)

    def stages(self) -> list[str]:
        return list(self._samples)

    def summary(self) -> list[StageSummary]:
        out = []
        for name, vals in self._samples.items():
            s = sorted(vals)
            out.append(
                StageSummary(
                    stage=name,
                    n=len(s),
                    mean_ms=sum(s) / len(s),
                    p50_ms=percentile(s, 50),
                    p95_ms=percentile(s, 95),
                    p99_ms=percentile(s, 99),
                    max_ms=s[-1],
                )
            )
        return out

    def end_to_end_ms(self) -> float:
        """Sum of per-stage mean latencies (stages run sequentially per frame)."""
        return sum(s.mean_ms for s in self.summary())


def joules_per_frame(avg_watts: float, frame_ms: float) -> float:
    """Energy for one frame at constant average power."""
    return avg_watts * frame_ms / 1000.0


def to_markdown(summaries: list[StageSummary], avg_watts: float | None = None) -> str:
    """Render a stage table; adds an energy column if ``avg_watts`` is given."""
    header = "| Stage | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |"
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    if avg_watts is not None:
        header += " mean mJ/frame |"
        rule += " --- |"
    rows = [header, rule]
    total_ms = 0.0
    for s in summaries:
        total_ms += s.mean_ms
        row = (
            f"| {s.stage} | {s.n} | {s.mean_ms:.2f} | {s.p50_ms:.2f} | "
            f"{s.p95_ms:.2f} | {s.p99_ms:.2f} | {s.max_ms:.2f} |"
        )
        if avg_watts is not None:
            row += f" {joules_per_frame(avg_watts, s.mean_ms) * 1000.0:.1f} |"
        rows.append(row)
    total = f"| **total (sum of means)** | | {total_ms:.2f} | | | | |"
    if avg_watts is not None:
        total += f" {joules_per_frame(avg_watts, total_ms) * 1000.0:.1f} |"
    rows.append(total)
    return "\n".join(rows)
