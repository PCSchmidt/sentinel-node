"""Fine-tune a small YOLO detector on VisDrone (aerial people and vehicles).

    python train.py --epochs 30

The first run downloads VisDrone (~2 GB) into ``<repo>/datasets`` (an absolute
path in a local data file; Ultralytics' global settings are not modified). The best
weights end up in ``out/runs/visdrone/weights/best.pt``.
"""

from __future__ import annotations

import argparse

from common import BASE_MODEL, IMGSZ, RUNS, local_visdrone_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="visdrone")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = local_visdrone_yaml()
    model = YOLO(BASE_MODEL)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=IMGSZ,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        device=0,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        plots=False,
    )


if __name__ == "__main__":
    main()
