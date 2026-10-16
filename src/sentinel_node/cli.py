"""Command-line entry: run the harness on synthetic data and print a stage table.

``python -m sentinel_node.cli --frames 200``

Numbers from this command measure the Python harness on synthetic stand-ins,
not the real node.
"""

from __future__ import annotations

import argparse

from .bench import to_markdown
from .pipeline import (
    CotPublisher,
    NearestNeighbourTracker,
    SyntheticDetector,
    SyntheticSource,
    run,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument(
        "--watts",
        type=float,
        default=None,
        help="average board power, to add an energy column (measure it; do not guess)",
    )
    args = ap.parse_args(argv)

    result = run(
        SyntheticSource(args.frames, n_targets=args.targets, fps=args.fps),
        SyntheticDetector(),
        NearestNeighbourTracker(),
        CotPublisher(origin_lat=38.9, origin_lon=-77.0),
    )
    print(f"{result.frames} frames, {result.bytes_sent} bytes published (synthetic stand-ins)\n")
    print(to_markdown(result.timer.summary(), avg_watts=args.watts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
