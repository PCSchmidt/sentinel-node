"""Capture → detect → track → publish pipeline with per-stage timing.

The stage interfaces (``FrameSource``, ``Detector``, ``Tracker``,
``Publisher``) are what the real Jetson implementations will fill. The
``Synthetic*`` and ``NearestNeighbourTracker`` classes below exist only to
exercise the harness end to end on a laptop. Timings measured with them say
nothing about real capture, inference or tracking cost.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from .bench import StageTimer
from .cot import GeoTrack, to_cot_xml

# Metres per degree of latitude (spherical approximation, fine for a node-local
# frame of a few kilometres).
M_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    # Ground-truth target positions in a local east/north frame (m). Real
    # frames would carry pixel data instead.
    truth_en: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Detection:
    east: float
    north: float
    score: float = 1.0


@dataclass(frozen=True)
class TrackOut:
    uid: str
    east: float
    north: float
    course_deg: float = 0.0
    speed_mps: float = 0.0


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


class Tracker(Protocol):
    def update(self, dets: list[Detection], t: float) -> list[TrackOut]: ...


class Publisher(Protocol):
    def publish(self, tracks: list[TrackOut], now: datetime) -> int:
        """Send ``tracks``; return bytes handed to the link."""
        ...


# ------------------------------------------------------------------ synthetic


class SyntheticSource:
    """Targets moving in straight lines in a local frame, at a fixed rate."""

    def __init__(self, n_frames: int, n_targets: int = 3, fps: float = 10.0, seed: int = 0):
        rng = random.Random(seed)
        self._n_frames = n_frames
        self._dt = 1.0 / fps
        self._targets = [
            (rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-15, 15), rng.uniform(-15, 15))
            for _ in range(n_targets)
        ]
        self._i = 0

    def read(self) -> Frame | None:
        if self._i >= self._n_frames:
            return None
        t = self._i * self._dt
        truth = tuple((e0 + ve * t, n0 + vn * t) for e0, n0, ve, vn in self._targets)
        frame = Frame(index=self._i, t=t, truth_en=truth)
        self._i += 1
        return frame


class SyntheticDetector:
    """Truth plus Gaussian noise, with missed detections."""

    def __init__(self, sigma_m: float = 2.0, pd: float = 0.9, seed: int = 1):
        self._rng = random.Random(seed)
        self._sigma = sigma_m
        self._pd = pd

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(e + self._rng.gauss(0, self._sigma), n + self._rng.gauss(0, self._sigma))
            for e, n in frame.truth_en
            if self._rng.random() < self._pd
        ]


class NearestNeighbourTracker:
    """Placeholder tracker: greedy nearest-neighbour with a distance gate.

    Not a fusion tracker. The real node tracker (see ``docs/design.md``) will
    replace it.
    """

    def __init__(self, gate_m: float = 15.0, max_age: int = 5):
        self._gate = gate_m
        self._max_age = max_age
        self._tracks: dict[str, dict] = {}
        self._next = 0

    def update(self, dets: list[Detection], t: float) -> list[TrackOut]:
        unused = list(dets)
        for uid, tr in list(self._tracks.items()):
            best = min(unused, key=lambda d: math.hypot(d.east - tr["e"], d.north - tr["n"]), default=None)
            if best is not None and math.hypot(best.east - tr["e"], best.north - tr["n"]) <= self._gate:
                dt = max(t - tr["t"], 1e-6)
                tr["ve"], tr["vn"] = (best.east - tr["e"]) / dt, (best.north - tr["n"]) / dt
                tr["e"], tr["n"], tr["t"], tr["age"] = best.east, best.north, t, 0
                unused.remove(best)
            else:
                tr["age"] += 1
                if tr["age"] > self._max_age:
                    del self._tracks[uid]
        for d in unused:
            self._tracks[f"sn-{self._next}"] = {"e": d.east, "n": d.north, "t": t, "ve": 0.0, "vn": 0.0, "age": 0}
            self._next += 1
        return [
            TrackOut(
                uid,
                tr["e"],
                tr["n"],
                course_deg=math.degrees(math.atan2(tr["ve"], tr["vn"])) % 360.0,
                speed_mps=math.hypot(tr["ve"], tr["vn"]),
            )
            for uid, tr in self._tracks.items()
        ]


# ------------------------------------------------------------------ publisher


class CotPublisher:
    """Encodes tracks as CoT events around a fixed geodetic origin."""

    def __init__(self, origin_lat: float, origin_lon: float, sink: Callable[[bytes], None] | None = None):
        self._lat0, self._lon0 = origin_lat, origin_lon
        self._sink = sink or (lambda _b: None)

    def publish(self, tracks: list[TrackOut], now: datetime) -> int:
        sent = 0
        for tr in tracks:
            lat = self._lat0 + tr.north / M_PER_DEG_LAT
            lon = self._lon0 + tr.east / (M_PER_DEG_LAT * math.cos(math.radians(self._lat0)))
            msg = to_cot_xml(
                GeoTrack(tr.uid, lat, lon, course_deg=tr.course_deg, speed_mps=tr.speed_mps), now
            )
            self._sink(msg)
            sent += len(msg)
        return sent


# --------------------------------------------------------------------- driver


@dataclass
class RunResult:
    frames: int
    bytes_sent: int
    timer: StageTimer


def run(source: FrameSource, detector: Detector, tracker: Tracker, publisher: Publisher) -> RunResult:
    """Run until the source is exhausted, timing each stage."""
    timer = StageTimer()
    frames = bytes_sent = 0
    while True:
        with timer.stage("capture"):
            frame = source.read()
        if frame is None:
            break
        with timer.stage("detect"):
            dets = detector.detect(frame)
        with timer.stage("track"):
            tracks = tracker.update(dets, frame.t)
        with timer.stage("publish"):
            bytes_sent += publisher.publish(tracks, datetime.now(timezone.utc))
        frames += 1
    return RunResult(frames=frames, bytes_sent=bytes_sent, timer=timer)
