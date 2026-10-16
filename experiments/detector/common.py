"""Shared paths and settings for the detector experiment."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
OUT = Path(__file__).resolve().parent / "out"
RUNS = OUT / "runs"
CALIB = OUT / "calib"

IMGSZ = 640
BASE_MODEL = "yolo11n.pt"
LOCAL_DATA = OUT / "visdrone_local.yaml"


def local_visdrone_yaml() -> str:
    """Write a copy of Ultralytics' VisDrone data file rooted inside this repo.

    The bundled file uses a relative ``path`` that Ultralytics resolves against
    its user-level ``datasets_dir`` setting, which is shared with every other
    project on the machine. Using an absolute ``path`` here keeps downloads in
    ``<repo>/datasets`` and leaves that global setting alone. The bundled file's
    download script is kept as is; it reads ``path`` from the same file.
    """
    import ultralytics

    src = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "VisDrone.yaml"
    text = src.read_text(encoding="utf-8")
    old = "path: VisDrone # dataset root dir"
    if old not in text:
        raise RuntimeError("unexpected VisDrone.yaml layout; update local_visdrone_yaml()")
    DATASETS.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA.write_text(
        text.replace(old, f"path: {(DATASETS / 'VisDrone').as_posix()} # dataset root dir (repo-local)"),
        encoding="utf-8",
    )
    return str(LOCAL_DATA)
