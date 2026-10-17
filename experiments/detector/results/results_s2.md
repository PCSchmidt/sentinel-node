| Precision | mAP50 | mAP50-95 | Δ mAP50-95 vs. baseline | Latency (ms/img) | Speedup | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.289 | 0.162 | baseline | 6.85 | 1.00x | 5.2 |
| TensorRT FP16 | 0.291 | 0.163 | +0.12 pts (+0.7%) | 1.29 | 5.31x | 8.2 |
| TensorRT INT8 (MinMax) | 0.278 | 0.154 | -0.80 pts (-5.0%) | 0.96 | 7.12x | 4.9 |
| TensorRT INT8 (Entropy) | 0.238 | 0.132 | -3.06 pts (-18.9%) | 0.92 | 7.48x | 4.9 |
