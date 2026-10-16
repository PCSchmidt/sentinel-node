import pytest

from sentinel_node.bench import (
    StageTimer,
    joules_per_frame,
    parse_tegrastats_power,
    percentile,
    to_markdown,
)


def test_percentile_basics():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(v, 0) == 1.0
    assert percentile(v, 50) == 3.0
    assert percentile(v, 100) == 5.0
    assert percentile(v, 25) == 2.0
    assert percentile([7.0], 99) == 7.0


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert percentile([0.0, 10.0], 95) == pytest.approx(9.5)


def test_percentile_validation():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_stage_timer_uses_injected_clock():
    ticks = iter([0.0, 0.010, 1.0, 1.030])
    t = StageTimer(clock=lambda: next(ticks))
    with t.stage("detect"):
        pass
    with t.stage("detect"):
        pass
    (s,) = t.summary()
    assert s.stage == "detect" and s.n == 2
    assert s.mean_ms == pytest.approx(20.0)
    assert s.max_ms == pytest.approx(30.0)
    assert s.p50_ms == pytest.approx(20.0)


def test_stage_timer_records_even_if_stage_raises():
    ticks = iter([0.0, 0.005])
    t = StageTimer(clock=lambda: next(ticks))
    with pytest.raises(RuntimeError):
        with t.stage("x"):
            raise RuntimeError
    assert t.summary()[0].n == 1


def test_end_to_end_is_sum_of_stage_means():
    t = StageTimer()
    t.record("a", 10.0)
    t.record("a", 20.0)
    t.record("b", 5.0)
    assert t.end_to_end_ms() == pytest.approx(20.0)


def test_energy():
    assert joules_per_frame(10.0, 50.0) == pytest.approx(0.5)


def test_markdown_table_has_energy_only_when_power_given():
    t = StageTimer()
    t.record("detect", 20.0)
    plain = to_markdown(t.summary())
    assert "mJ/frame" not in plain
    with_power = to_markdown(t.summary(), avg_watts=10.0)
    assert "mJ/frame" in with_power and "200.0" in with_power  # 10 W * 20 ms


def test_parse_tegrastats_power():
    line = "RAM 2000/7620MB CPU [10%@1510] GPU 5% VDD_IN 5209mW/5209mW VDD_CPU_GPU_CV 800mW/800mW"
    assert parse_tegrastats_power(line) == pytest.approx(5.209)
    assert parse_tegrastats_power("no power field here") is None
