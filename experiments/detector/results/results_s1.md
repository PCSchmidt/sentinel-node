| Precision | mAP50 | mAP50-95 | Δ mAP50-95 vs. baseline | Latency (ms/img) | Speedup | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.291 | 0.163 | baseline | 6.88 | 1.00x | 5.2 |
| TensorRT FP16 | 0.294 | 0.165 | +0.16 pts (+1.0%) | 1.03 | 6.70x | 8.4 |
| TensorRT INT8 (MinMax) | 0.278 | 0.154 | -0.92 pts (-5.7%) | 0.99 | 6.98x | 5.1 |
| TensorRT INT8 (Entropy) | 0.242 | 0.132 | -3.13 pts (-19.2%) | 0.92 | 7.52x | 4.8 |
