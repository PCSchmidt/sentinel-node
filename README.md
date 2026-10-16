# Sentinel Node

[![CI](https://github.com/PCSchmidt/sentinel-node/actions/workflows/ci.yml/badge.svg)](https://github.com/PCSchmidt/sentinel-node/actions/workflows/ci.yml)

An attritable edge sensor: a low-cost node that detects people, vehicles and small drones, tracks them locally, and publishes **tracks, not video** over a constrained link. Target hardware is a Jetson Orin Nano on a 7-25 W budget.

**Status: scaffold, no device results yet.** The measurement harness, Cursor-on-Target encoding and bandwidth governor are implemented and unit-tested on a laptop. Nothing has run on a Jetson, there is no real detector, camera or GPS yet, and no latency, energy or accuracy numbers exist. See the [roadmap](docs/roadmap.md).

Part of [physical-ai-portfolio](https://github.com/PCSchmidt/physical-ai-portfolio). The node tracker is intended to reuse ideas from [FuseTrack](https://github.com/PCSchmidt/fusetrack).

## Problem

A forward sensor has little power, a thin link and no operator watching video. It has to decide locally what is there and send only what the link can carry. The interesting question is the trade curve: how much accuracy, latency and battery each optimisation costs. That curve is the deliverable, not the demo.

## What exists today

| Module | What it does | Tested |
| --- | --- | --- |
| `cot` | Encodes a track as a CoT 2.0 event (point + course/speed) | Structure, UTC times, range checks, size under 600 B |
| `governor` | Chooses what to publish (tracks + thumbnails, tracks, reduced-rate tracks, nothing) from measured link capacity, with hysteresis | Exact thresholds, immediate degrade, no flapping |
| `bench` | Per-stage timers with p50/p95/p99, energy per frame from average power, `tegrastats` power parsing | Percentiles, injected clock, markdown table |
| `pipeline` | capture → detect → track → publish driver with stage interfaces | End to end on synthetic stand-ins |

The `Synthetic*` classes and the nearest-neighbour tracker exist only to exercise the harness. Timings from them say nothing about the real node.

## Planned measurements (not yet done)

1. Latency per stage and joules per frame at 7 W, 15 W and 25 W power modes.
2. Detection mAP before and after INT8 quantisation, with the calibration set documented.
3. Tracking MOTA/HOTA on recorded sequences, and track continuity under dropped frames.
4. Tracks delivered per kilobyte at simulated 9.6 kbps, 64 kbps and 1 Mbps.
5. A 24-hour soak test.

## Run the harness

```sh
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m sentinel_node.cli --frames 300
```

## Scope

Passive sensing, tracking and reporting only. Public data and unclassified scenarios. No weapons, targeting or engagement functionality.

## License

MIT. See [LICENSE](LICENSE).
