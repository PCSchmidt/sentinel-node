# Detector experiment (milestone 0.3a): FP32 vs. FP16 vs. INT8

How much accuracy does quantisation cost a small aerial detector, and how much latency does it buy? This runs on a desktop NVIDIA GPU, so it needs no Jetson. It measures the **accuracy** cost of INT8, which indicates what to expect on the Jetson but does not guarantee it: the engines are rebuilt there with a different TensorRT version and may quantise differently. It does not measure Jetson latency or power (milestone 0.4).

**Status: first results in** (single run, seed 0). See Results below; they are from a desktop GPU and are not Jetson numbers.

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

Fine-tuned YOLO11n on VisDrone, 30 epochs (final training-time validation: mAP50 0.299, mAP50-95 0.165). The table below uses `evaluate.py`'s settings (batch 1, full-precision validator), which score slightly lower than the training-time validation; the same settings are used for every row. RTX A4500 Laptop GPU, TensorRT 10.16, PyTorch 2.11 (CUDA 12.8). Raw files are in [results/](results/).

| Precision | mAP50 | mAP50-95 | Δ mAP50-95 vs. baseline | Latency (ms/img) | Speedup | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.290 | 0.162 | baseline | 6.82 | 1.00x | 5.2 |
| TensorRT FP16 | 0.290 | 0.162 | +0.05 pts (+0.3%) | 1.05 | 6.53x | 8.1 |
| TensorRT INT8 | 0.278 | 0.152 | -0.92 pts (-5.7%) | 1.52 | 4.49x | 4.9 |

What this shows:

- **FP16 is free here.** Same accuracy to within noise (+0.05 points) and 6.5x lower latency than PyTorch. Part of that gap is TensorRT versus eager PyTorch overhead on a small model, not purely precision, so it overstates what FP16 alone buys.
- **INT8 cost accuracy and did not buy speed.** mAP50-95 fell 0.92 points (5.7% relative), and INT8 was *slower* than FP16 (1.52 ms vs 1.05 ms). Its only advantage on this setup is a smaller file (4.9 MB vs 8.1 MB for the FP16 engine).
- **I have not profiled why INT8 is slower.** Plausible causes, untested: a very small model at batch 1 is limited by kernel launch and memory traffic rather than arithmetic; the mixed-precision graph (head kept in higher precision) adds quantise/dequantise steps; and the INT8 kernels may not be selected on this GPU for these layer shapes.
- **Implication for the Jetson:** do not assume INT8 is the fast path. The Jetson milestone should measure FP16 and INT8 on the device, with power, and report whichever wins on joules per frame.

## Limitations of these results

- One training run and one calibration sample (512 images, seed 0). The 0.05-point FP16 difference is noise; the 0.92-point INT8 drop is probably real in sign but its size depends on the calibration set and method. MinMax calibration is sensitive to outliers, and a percentile or entropy calibrator may lose less accuracy; that was not tried.
- Latency is median model-inference time for one image on an RTX A4500 laptop GPU, not the target device, and excludes pre- and post-processing. It was measured on a random-noise image, so postprocessing cost is not represented.
- Latency came from a laptop GPU that may throttle under load; the 300-run median reduces but does not remove that.

## Other limitations

- VisDrone is aerial people and vehicles, not small drones in the air; a drone-specific detection dataset is not chosen yet.
- Ultralytics is AGPL-3.0 licensed; this repository uses it as a tool and does not redistribute it.
