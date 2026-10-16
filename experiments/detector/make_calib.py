"""Pick a held-out INT8 calibration set from the *training* images.

Ultralytics calibrates INT8 on the ``val`` split of the data file by default.
Scoring on that same split would let calibration see the evaluation images, so
this script writes a separate data file whose ``val`` is a random sample of
training images. The real validation split is never used for calibration.

    python make_calib.py --n 512
"""

from __future__ import annotations

import argparse
import random

from common import CALIB, local_visdrone_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(local_visdrone_yaml())
    train_dir = data["train"]
    from pathlib import Path

    images = sorted(p for p in Path(train_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < args.n:
        raise SystemExit(f"only {len(images)} training images found under {train_dir}")
    sample = random.Random(args.seed).sample(images, args.n)

    CALIB.mkdir(parents=True, exist_ok=True)
    listing = CALIB / "calib.txt"
    listing.write_text("\n".join(p.as_posix() for p in sample) + "\n", encoding="utf-8")

    names = data["names"]
    lines = [f"path: {CALIB.as_posix()}", "train: calib.txt", "val: calib.txt", "names:"]
    lines += [f"  {i}: {n}" for i, n in names.items()]
    (CALIB / "calib.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {listing} ({len(sample)} train images, seed {args.seed}) and calib.yaml")


if __name__ == "__main__":
    main()
