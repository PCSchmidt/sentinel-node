# Detector experiment (milestone 0.3a): FP32 vs. FP16 vs. INT8

How much accuracy does quantisation cost a small aerial detector, and how much latency does it buy? This runs on a desktop NVIDIA GPU, so it needs no Jetson. It measures the **accuracy** cost of INT8, which indicates what to expect on the Jetson but does not guarantee it: the engines are rebuilt there with a different TensorRT version and may quantise differently. It does not measure Jetson latency or power (milestone 0.4).

**Status: results for the seed-0 model, calibration study and latency re-measurement are in; two extra training seeds are still running.** Everything is from a desktop GPU and is not a Jetson number. An earlier version of this page reported INT8 as slower than FP16; that was wrong, see the correction below.

## Method

1. **Model:** YOLO11n (Ultralytics), COCO-pretrained, fine-tuned on VisDrone2019-DET (aerial views; classes include pedestrian, people, car, van, truck, bus, motor). 640 px input, seed 0.
2. **Calibration set:** 512 images sampled at random (seed 0) from the **training** split, written by `make_calib.py`. Ultralytics would otherwise calibrate on the `val` split, which is also what we score on, so calibration would see the evaluation images.
3. **Engines:** built from the same checkpoint, batch 1, static 640x640. INT8 uses TensorRT's MinMax calibrator via Ultralytics; per its source, the head's final Sigmoid is kept in higher precision to preserve confidence scores, so this is mixed-precision INT8, not every layer.
4. **Metrics:** mAP50 and mAP50-95 on the VisDrone validation split (Ultralytics validator, default confidence threshold); latency is the median of 300 single-image predictions after 30 warm-up runs, model inference only (no pre/post-processing); size is the file on disk.
5. **Software:** TensorRT 10.x (the same major version as recent JetPack 6.x releases; the exact minor differs) with PyTorch built for CUDA 12.8.

## Run

```sh
# GPU environment (Python 3.12). Not installed by `pip install -e .`
python -m venv .venv-gpu
.venv-gpu/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv-gpu/Scripts/python -m pip install ultralytics onnx "tensorrt-cu12<11"

cd experiments/detector
python train.py --epochs 30          # downloads VisDrone (~2 GB) on first run
python make_calib.py --n 512
python evaluate.py --weights out/runs/visdrone/weights/best.pt
```

TensorRT 11 removed the classic INT8 calibrator, so Ultralytics routes it through NVIDIA ModelOpt instead. That is a different quantisation path from JetPack's TensorRT 10.x, which is why this experiment pins `tensorrt-cu12<11`.

## Results

Fine-tuned YOLO11n on VisDrone, 30 epochs (final training-time validation: mAP50 0.299, mAP50-95 0.165). The table uses `evaluate.py`'s settings (batch 1, full-precision validator), which score slightly lower than the training-time validation; the same settings apply to every row. RTX A4500 Laptop GPU, TensorRT 10.16, PyTorch 2.11 (CUDA 12.8). Raw files are in [results/](results/).

| Precision | mAP50 | mAP50-95 | Δ mAP50-95 vs. baseline | Latency (ms/img) | Speedup | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.290 | 0.162 | baseline | 6.59 | 1.00x | 5.2 |
| TensorRT FP16 | 0.290 | 0.162 | +0.05 pts (+0.3%) | 1.12 | 5.87x | 8.1 |
| TensorRT INT8 (MinMax) | 0.278 | 0.152 | -0.92 pts (-5.7%) | 1.07 | 6.17x | 4.9 |

Latency is the median of 5 interleaved rounds (see below). The FP32-to-TensorRT gap partly reflects eager PyTorch overhead on a tiny model, not precision alone, so read the FP16-vs-INT8 comparison, not the FP32 one, for the effect of quantisation.

### Correction: INT8 is not slower than FP16

The first run measured INT8 at 1.52 ms and FP16 at 1.05 ms, and this page said INT8 was slower. That did not reproduce. Re-measuring the same engines in interleaved rounds, the original INT8 engine ran at 0.92-1.14 ms and FP16 at 1.05-1.33 ms, with INT8 faster than FP16 in every round. The 1.52 ms reading was an outlier of unknown cause. The GPU throttles during a run (SM clock fell from 1785 to 1305 MHz as it heated), which makes absolute latency drift upward by about 15% together for all engines, but I have not shown that throttling explains the outlier. The first-pass numbers are kept in `results/results_first_pass.json`.

The reliable way to compare engines on this machine is within the same round. Median same-round latency relative to FP16:

| Engine | Batch 1 | Batch 8 |
| --- | --- | --- |
| TensorRT FP16 | 1.00 | 1.00 |
| TensorRT INT8, MinMax calibration | 0.92 (faster in every round) | 0.73 (faster in every round) |
| TensorRT INT8, Entropy calibration | 0.83 | 0.97 |

At batch 8 the MinMax INT8 engine ran at about 0.53 ms/img against 0.82 ms/img for FP16 (medians; ranges overlap across rounds because of throttling, hence the same-round ratios).

### INT8 calibration: algorithm and sample

Same seed-0 weights; each row is a separate INT8 build calibrated on a different random 512-image sample of the training split. I confirmed in the TensorRT build log that the first three builds used `MinMaxCalibration` and the last three `EntropyCalibration2`. Entropy calibration needed a narrow patch of an Ultralytics internal (`calibration.py`), which is described there.

| INT8 calibration | Calibration sample (seed) | mAP50-95 | Δ vs. FP32 (pts) |
| --- | --- | --- | --- |
| MinMax | 0 | 0.1539 | -0.78 |
| MinMax | 1 | 0.1534 | -0.83 |
| MinMax | 2 | 0.1542 | -0.75 |
| Entropy | 0 | 0.1198 | -4.19 |
| Entropy | 1 | 0.1387 | -2.30 |
| Entropy | 2 | 0.1378 | -2.39 |

MinMax: mean -0.79 pts, std 0.04 over the three samples. Entropy: mean -2.96 pts, std 1.06. The first MinMax build on calibration sample 0 scored -0.92 pts and a rebuild with the same sample scored -0.78, so TensorRT builds differ by roughly 0.14 points even with identical inputs.

### What this shows

- **FP16 is free here:** same accuracy to within noise.
- **INT8 with MinMax calibration costs about 0.8-0.9 mAP50-95 points (about 5%)** and is stable across calibration samples; in return it is about 8% faster than FP16 at batch 1 and about 27% faster at batch 8 on this GPU. A modest return at batch 1.
- **Entropy calibration was worse on this model:** about 3 points lost, with much more variation between calibration samples, and no faster than FP16 at batch 8. I did not investigate why; the measurement is what it is.
- **The earlier hypothesis that INT8 is launch-bound at batch 1 is not needed:** INT8 is faster, just by little at batch 1, and the gap grows with batch size, which is what a partly launch- or memory-bound small model would show. I have not profiled it.
- **For the Jetson:** measure FP16 and INT8 on the device with clocks fixed, repeat and interleave the measurements, and decide on joules per frame, not latency alone.

Training-seed results (two more fine-tunes) are running and will be added here.

## Limitations of these results

- One trained model (seed 0) for the calibration and latency studies; the other training seeds are pending. Mean-and-spread figures above are over calibration samples, not over trained models.
- Latency is from a laptop GPU that throttles. Absolute values drift by about 15% during a run; only same-round comparisons are trusted. Latency is model inference only, on a random-noise image, and is not Jetson latency.
- The MinMax and Entropy comparison uses Ultralytics' mixed-precision graph (the head's final Sigmoid is kept in higher precision), which is a specific choice. Other calibrators, such as percentile, were not tried because Ultralytics does not expose them for this TensorRT path.
- Build-to-build variation (about 0.14 points for the same inputs) is of the same order as the differences between the three MinMax samples, so the small differences among those rows should not be read as meaningful.

## Other limitations

- VisDrone is aerial people and vehicles, not small drones in the air; a drone-specific detection dataset is not chosen yet.
- Ultralytics is AGPL-3.0 licensed; this repository uses it as a tool and does not redistribute it.
