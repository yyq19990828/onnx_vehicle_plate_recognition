"""
The infer_onnx package provides classes for ONNX-based inference.
"""

# Import classes from submodules to make them accessible at the package level,
# e.g., so you can do `from infer_onnx import DetONNX`.
from .det_onnx import DetONNX
from .ocr_onnx import ColorLayerONNX, OCRONNX

# This makes `from infer_onnx import *` behave nicely, exporting only these names.
__all__ = ['DetONNX', 'ColorLayerONNX', 'OCRONNX']