"""
The infer_onnx package provides classes for ONNX-based inference.
"""

# Import classes from new unified module
from .yolo_models import (
    YoloOnnx, 
    RTDETROnnx, 
    RFDETROnnx, 
    DetONNX,  # 向后兼容
    YoloRTDETROnnx,  # 向后兼容
    create_detector,
    DatasetEvaluator
)
from .ocr_onnx import ColorLayerONNX, OCRONNX

# This makes `from infer_onnx import *` behave nicely, exporting only these names.
__all__ = [
    'YoloOnnx', 
    'RTDETROnnx', 
    'RFDETROnnx', 
    'DetONNX',  # 向后兼容
    'YoloRTDETROnnx',  # 向后兼容
    'ColorLayerONNX', 
    'OCRONNX',
    'create_detector',
    'DatasetEvaluator'
]