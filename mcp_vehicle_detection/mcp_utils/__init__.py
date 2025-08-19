"""
MCP工具模块
包含验证、图片处理、统一检测等功能
"""

from .validation import (
    validate_image_path, validate_image_paths, validate_video_path,
    validate_confidence_threshold, sanitize_filename, validate_image_input,
    is_base64_image, decode_base64_image
)

from .image_processor import (
    ImageProcessor, process_image_input, cleanup_temp_files, get_image_processor
)

from .unified_detection import (
    UnifiedVehicleDetector, DetectionToolFactory
)

__all__ = [
    'validate_image_path', 'validate_image_paths', 'validate_video_path',
    'validate_confidence_threshold', 'sanitize_filename', 'validate_image_input',
    'is_base64_image', 'decode_base64_image',
    'ImageProcessor', 'process_image_input', 'cleanup_temp_files', 'get_image_processor',
    'UnifiedVehicleDetector', 'DetectionToolFactory'
]