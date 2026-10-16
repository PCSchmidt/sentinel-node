import pytest

from sentinel_node.detector_report import (
    DetectorResult,
    compare,
    load_results,
    save_results,
    to_markdown,
)

FP32 = DetectorResult("PyTorch FP32", map50=0.40, map50_95=0.25, latency_ms=10.0, size_mb=5.0)
INT8 = DetectorResult("TensorRT INT8", map50=0.39, map50_95=0.24, latency_ms=2.0, size_mb=2.5)


def test_compare_accuracy_cost_and_speedup():
    c = compare(INT8, FP32)
    assert c.map50_95_delta == pytest.approx(-0.01)
    assert c.map50_95_rel_pct == pytest.approx(-4.0)
    assert c.speedup == pytest.approx(5.0)


def test_compare_against_self_is_neutral():
    c = compare(FP32, FP32)
    assert c.map50_95_delta == 0.0 and c.speedup == 1.0


def test_compare_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        compare(INT8, DetectorResult("bad", 0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        compare(DetectorResult("bad", 0.1, 0.1, 0.0), FP32)


def test_markdown_table():
    md = to_markdown([FP32, INT8])
    lines = md.splitlines()
    assert len(lines) == 4
    assert "baseline" in lines[2] and "1.00x" in lines[2]
    assert "-1.00 pts (-4.0%)" in lines[3]
    assert "5.00x" in lines[3]


def test_markdown_handles_missing_size_and_requires_results():
    r = DetectorResult("x", 0.1, 0.1, 1.0)
    assert "n/a" in to_markdown([r])
    with pytest.raises(ValueError):
        to_markdown([])


def test_round_trip_json(tmp_path):
    path = tmp_path / "r.json"
    save_results(path, [FP32, INT8], meta={"gpu": "test"})
    results, meta = load_results(path)
    assert results == [FP32, INT8]
    assert meta == {"gpu": "test"}


def test_mean_std():
    from sentinel_node.detector_report import mean_std

    assert mean_std([2.0, 4.0, 6.0]) == pytest.approx((4.0, 2.0))
    assert mean_std([5.0]) == (5.0, 0.0)
    with pytest.raises(ValueError):
        mean_std([])
