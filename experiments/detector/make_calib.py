"""Pick a held-out INT8 calibration set from the *training* images.

Ultralytics calibrates INT8 on the ``val`` split of the data file by default.
Scoring on that split would let calibration see the evaluation images, so
this script writes a separate data file whose ``val`` is a random sample of
training images. The real validation split is never used for calibration.

    python make_calib.py --n 512 --seed 0
    python make_calib.py --n 512 --seed 1 --out out/calib/seed1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import CALIB, local_visdrone_yaml


def make_calib(n: int, seed: int, out: Path) -> Path:
    """Write ``calib.txt`` and ``calib.yaml`` into ``out``; return the yaml path."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(local_visdrone_yaml())
    train_dir = data["train"]
    images = sorted(p for p in Path(train_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < n:
        raise SystemExit(f"only {len(images)} training images found under {train_dir}")
    sample = random.Random(seed).sample(images, n)

    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.txt").write_text("\n".join(p.as_posix() for p in sample) + "\n", encoding="utf-8")
    lines = [f"path: {out.as_posix()}", "train: calib.txt", "val: calib.txt", "names:"]
    lines += [f"  {i}: {name}" for i, name in data["names"].items()]
    yaml_path = out / "calib.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=CALIB)
    args = ap.parse_args()
    path = make_calib(args.n, args.seed, args.out)
    print(f"wrote {path} ({args.n} train images, seed {args.seed})")


if __name__ == "__main__":
    main()
