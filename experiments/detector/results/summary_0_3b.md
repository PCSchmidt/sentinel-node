### INT8 calibration algorithm and sample (seed-0 model)

| INT8 calibration | Calibration sample (seed) | mAP50-95 | Δ vs. FP32 (pts) |
| --- | --- | --- | --- |
| minmax | 0 | 0.1539 | -0.78 |
| minmax | 1 | 0.1534 | -0.83 |
| minmax | 2 | 0.1542 | -0.75 |
| entropy | 0 | 0.1198 | -4.19 |
| entropy | 1 | 0.1387 | -2.30 |
| entropy | 2 | 0.1378 | -2.39 |

| INT8 calibration | n samples | Mean Δ vs. FP32 (pts) | Std (pts) | Range (pts) |
| --- | --- | --- | --- | --- |
| minmax | 3 | -0.79 | 0.04 | -0.83 to -0.75 |
| entropy | 3 | -2.96 | 1.06 | -4.19 to -2.30 |

### Across training seeds

| Training seed | FP32 mAP50-95 | TensorRT FP16 Δ (pts) | TensorRT INT8 Δ (pts) | TensorRT INT8 (Entropy) Δ (pts) |
| --- | --- | --- | --- | --- |
| 0 | 0.1617 | +0.05 | -0.92 | n/a |
| 1 | 0.1633 | +0.16 | -0.92 | -3.13 |
| 2 | 0.1623 | +0.12 | -0.80 | -3.06 |
| **mean ± std** | 0.1624 ± 0.0008 | +0.11 ± 0.05 (n=3) | -0.88 ± 0.07 (n=3) | -3.10 ± 0.05 (n=2) |

### Latency, interleaved rounds (seed-0 model)

| Engine | Batch | Median ms/img | Range (ms) | Median same-round ratio vs. FP16 |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 1 | 6.59 | 6.46-6.67 | 5.87 |
| FP16 (original build) | 1 | 1.12 | 1.05-1.33 | 1.00 |
| INT8 MinMax (original build) | 1 | 1.07 | 0.92-1.14 | 0.92 |
| INT8 MinMax (rebuild, calib seed 0) | 1 | 1.13 | 0.95-1.20 | 0.93 |
| INT8 Entropy (calib seed 0) | 1 | 1.00 | 0.86-1.10 | 0.83 |
| FP16 batch 8 | 8 | 0.82 | 0.57-0.82 | 1.00 |
| INT8 MinMax batch 8 | 8 | 0.53 | 0.44-0.61 | 0.73 |
| INT8 Entropy batch 8 | 8 | 0.79 | 0.60-0.80 | 0.97 |
