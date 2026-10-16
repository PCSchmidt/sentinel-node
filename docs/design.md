# Design

## Data flow

```text
camera (CSI) ─▶ capture ─▶ detect (TensorRT INT8) ─▶ track ─▶ governor ─▶ publish (CoT over MQTT / UDP)
                  │             │                        │          ▲
                  └─────────────┴──── per-stage timers ──┴──────────┘
                                       + power sampling
```

Each stage is a small interface in `pipeline.py` (`FrameSource`, `Detector`, `Tracker`, `Publisher`). The laptop harness uses synthetic implementations; the Jetson build swaps in real ones without touching the driver or the timing code.

## Decisions made

- **Publish tracks, not video.** A CoT event is a few hundred bytes (a test pins it under 600 B). A video stream is orders of magnitude more.
- **Stage timers measure wall-clock time per stage**, summarised as percentiles. End-to-end latency is reported as the sum of mean stage latencies, which is valid only because stages run sequentially per frame; a pipelined build would need per-frame timestamps instead.
- **Energy is average power times latency.** The power figure must be measured (board sensor or an external meter), not assumed. `parse_tegrastats_power` reads `VDD_IN` from `tegrastats` output; the field name still needs checking against real output on the device.
- **The governor degrades immediately and upgrades with margin.** A link that can't carry the current level must shed load now; flapping near a threshold is worse than staying one level low.

## Open decisions

- **Tracker.** Options are a C++ port of the FuseTrack GNN/IMM ideas, calling the Rust FuseTrack crate through a thin FFI or sidecar process, or a Python tracker for the first measurements. The choice depends on what the first latency table shows; the placeholder nearest-neighbour tracker is not a candidate.
- **Detector.** A YOLO-class model fine-tuned on a drone/vehicle/person set, exported ONNX → TensorRT. Dataset choice is not made.
- **Geo-registration.** Turning pixel detections into lat/lon needs a camera calibration and a GPS fix. The current publisher assumes a local east/north frame.
- **Transport.** MQTT and UDP multicast are both in scope; the encoder is transport-agnostic and neither is implemented.

## Known limitations

- Nothing has run on a Jetson.
- The local-frame to lat/lon conversion is a flat-earth approximation suitable for a few kilometres around the node.
- The CoT output uses a minimal subset (point and track detail) and has not been checked against a real ATAK/WinTAK client.
