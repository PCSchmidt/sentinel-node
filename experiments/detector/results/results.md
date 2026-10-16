| Precision | mAP50 | mAP50-95 | Δ mAP50-95 vs. baseline | Latency (ms/img) | Speedup | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.290 | 0.162 | baseline | 6.82 | 1.00x | 5.2 |
| TensorRT FP16 | 0.290 | 0.162 | +0.05 pts (+0.3%) | 1.05 | 6.53x | 8.1 |
| TensorRT INT8 | 0.278 | 0.152 | -0.92 pts (-5.7%) | 1.52 | 4.49x | 4.9 |
