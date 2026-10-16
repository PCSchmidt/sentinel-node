"""Switch Ultralytics' TensorRT INT8 calibration algorithm.

Ultralytics hard-codes TensorRT's MinMax calibrator for GPU engines (entropy is
used only for DLA). Its calibrator class is defined inside ``onnx2engine``, so it
cannot be subclassed. It does, however, read the algorithm from
``trt.CalibrationAlgoType.MINMAX_CALIBRATION`` at construction, and the same
class works with entropy calibration (that is exactly what its DLA path does).
``entropy_calibration()`` temporarily points that name at the entropy-2
algorithm. This is a deliberate, narrow patch of a third-party internal and
could break with other Ultralytics versions; verify in the TensorRT build log
that the calibration cache is named ``...EntropyCalibration2``.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def entropy_calibration():
    import tensorrt as trt

    real = trt.CalibrationAlgoType

    class _Shim:
        MINMAX_CALIBRATION = real.ENTROPY_CALIBRATION_2  # the name Ultralytics reads
        ENTROPY_CALIBRATION_2 = real.ENTROPY_CALIBRATION_2

    trt.CalibrationAlgoType = _Shim
    try:
        yield
    finally:
        trt.CalibrationAlgoType = real
