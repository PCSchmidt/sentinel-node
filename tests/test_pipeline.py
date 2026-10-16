import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest

from sentinel_node.pipeline import (
    CotPublisher,
    Detection,
    NearestNeighbourTracker,
    SyntheticDetector,
    SyntheticSource,
    TrackOut,
    run,
)


def test_synthetic_source_yields_requested_frames_then_none():
    src = SyntheticSource(n_frames=3, n_targets=2)
    frames = [src.read() for _ in range(3)]
    assert all(f is not None and len(f.truth_en) == 2 for f in frames)
    assert src.read() is None


def test_synthetic_source_targets_move():
    src = SyntheticSource(n_frames=2, n_targets=1, fps=1.0, seed=3)
    a, b = src.read(), src.read()
    assert a.truth_en != b.truth_en


def test_detector_is_deterministic_for_a_seed():
    frame = SyntheticSource(1, 3).read()
    assert SyntheticDetector(seed=5).detect(frame) == SyntheticDetector(seed=5).detect(frame)


def test_tracker_keeps_ids_for_separated_moving_targets():
    tr = NearestNeighbourTracker(gate_m=15.0)
    ids = None
    for k in range(20):
        t = float(k)
        dets = [Detection(10.0 * t, 0.0), Detection(0.0, 500.0 - 5.0 * t)]
        out = tr.update(dets, t)
        if k == 2:
            ids = sorted(o.uid for o in out)
    assert ids is not None
    assert sorted(o.uid for o in out) == ids


def test_tracker_estimates_course_and_speed():
    tr = NearestNeighbourTracker()
    for k in range(5):
        out = tr.update([Detection(10.0 * k, 0.0)], float(k))
    (o,) = out
    assert o.speed_mps == pytest.approx(10.0)
    assert o.course_deg == pytest.approx(90.0)  # due east


def test_tracker_drops_a_lost_track_after_max_age():
    tr = NearestNeighbourTracker(max_age=2)
    tr.update([Detection(0.0, 0.0)], 0.0)
    for k in range(1, 5):
        out = tr.update([], float(k))
    assert out == []


def test_publisher_converts_local_offsets_to_lat_lon():
    sent = []
    pub = CotPublisher(38.0, -77.0, sink=sent.append)
    n = pub.publish([TrackOut("a", east=0.0, north=1113.2)], datetime(2026, 1, 1, tzinfo=timezone.utc))
    (msg,) = sent
    assert n == len(msg)
    pt = ET.fromstring(msg).find("point")
    # 1113.2 m north is ~0.01 degrees of latitude.
    assert float(pt.get("lat")) == pytest.approx(38.01, abs=1e-4)
    assert float(pt.get("lon")) == pytest.approx(-77.0, abs=1e-6)


def test_end_to_end_run_times_all_four_stages():
    pub = CotPublisher(38.9, -77.0)
    res = run(SyntheticSource(50, 3), SyntheticDetector(), NearestNeighbourTracker(), pub)
    assert res.frames == 50
    assert res.bytes_sent > 0
    stages = {s.stage: s for s in res.timer.summary()}
    assert set(stages) == {"capture", "detect", "track", "publish"}
    # capture is also timed on the final empty read, so it has one extra sample
    assert stages["detect"].n == 50 and stages["capture"].n == 51
