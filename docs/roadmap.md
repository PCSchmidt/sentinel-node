# Roadmap

| Milestone | Deliverable | Needs hardware | Status |
| --- | --- | --- | --- |
| 0.1 | Harness: CoT encoder, bandwidth governor, stage timers, pipeline driver, CI | No | Done |
| 0.2 | Validate CoT output in a real ATAK/WinTAK client; add UDP/MQTT publishers | No | Planned |
| 0.3 | Capture on Jetson: CSI camera via GStreamer/Argus, zero-copy to GPU; capture latency measured | Jetson, camera | Planned |
| 0.3a | Detector fine-tune and ONNX → TensorRT INT8 accuracy (mAP before/after) on a desktop NVIDIA GPU; no Jetson needed | No (laptop RTX GPU) | Done (single run) |
| 0.3b | Other INT8 calibration methods and multiple seeds, to see whether the INT8 accuracy loss shrinks | No (laptop RTX GPU) | Calibration and latency studies done; extra training seeds running |
| 0.4 | Rebuild engines on the Jetson; latency table and joules per frame at 7/15/25 W | Jetson | Planned |
| 0.5 | On-node tracker, geo-registered with GPS and a calibrated camera; CoT into ATAK/WinTAK | Jetson, GPS | Planned |
| 0.6 | Ops layer: systemd services, watchdog, health telemetry, read-only root filesystem, power-state machine | Jetson | Planned |
| 0.7 | 24-hour soak test; link results at 9.6 kbps, 64 kbps, 1 Mbps | Jetson | Planned |
| Stretch | Two nodes cross-cueing; feed FuseTrack | 2 × Jetson | Planned |

A milestone is done only when its README section leads with measured results.
