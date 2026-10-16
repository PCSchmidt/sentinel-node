# Sentinel Node

[![CI](https://github.com/PCSchmidt/sentinel-node/actions/workflows/ci.yml/badge.svg)](https://github.com/PCSchmidt/sentinel-node/actions/workflows/ci.yml)

An attritable edge sensor: a low-cost node that detects people, vehicles and small drones, tracks them locally, and publishes **tracks, not video** over a constrained link. Target hardware is a Jetson Orin Nano on a 7-25 W budget.

**Status: scaffold plus a first detector result.** The measurement harness, Cursor-on-Target encoding and bandwidth governor are implemented and unit-tested on a laptop, and a YOLO11n detector has been fine-tuned on VisDrone and compared across FP32, FP16 and INT8 on a desktop GPU (below). Nothing has run on a Jetson, and there is no camera, GPS or on-node tracker yet, so no device latency or energy numbers exist. See the [roadmap](docs/roadmap.md).

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

## First result: detector precision on a desktop GPU

YOLO11n fine-tuned on VisDrone (aerial people and vehicles), 640 px, evaluated on the VisDrone validation split with INT8 calibrated on 512 held-out training images. RTX A4500 Laptop GPU, TensorRT 10.16.

| Precision | mAP50-95 | Δ vs. FP32 | Latency, batch 1 (ms/img) | Size (MB) |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.162 | baseline | 6.59 | 5.2 |
| TensorRT FP16 | 0.162 | +0.05 pts | 1.12 | 8.1 |
| TensorRT INT8 (MinMax) | 0.152 | -0.92 pts (-5.7%) | 1.07 | 4.9 |

FP16 cost nothing in accuracy. MinMax INT8 lost about 0.8-0.9 points of mAP50-95 (stable across three calibration samples) and ran about 8% faster than FP16 at batch 1 and about 27% faster at batch 8; entropy calibration lost about 3 points. An earlier version of this page said INT8 was slower than FP16; that came from a single outlier latency reading and did not reproduce (details and correction in the experiment README). Latency on this throttling laptop GPU drifts, so engines are compared within the same measurement round. Not Jetson numbers; two more training seeds are still running. See [experiments/detector](experiments/detector/README.md).

## Planned measurements (not yet done)

1. Latency per stage and joules per frame at 7 W, 15 W and 25 W power modes, on the Jetson (including FP16 vs. INT8).
2. Detection mAP after INT8 on the device build, and with other calibration methods.
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
