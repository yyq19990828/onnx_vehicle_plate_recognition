"""
输入验证工具函数
"""

import os
import base64
import tempfile
from typing import List, Union, Tuple
from pathlib import Path
import re


def validate_image_path(image_path: str) -> bool:
    """
    验证图片路径是否有效
    
    Args:
        image_path: 图片路径
        
    Returns:
        是否有效
    """
    if not image_path or not isinstance(image_path, str):
        return False
    
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return False
    
    # 检查文件扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return path.suffix.lower() in valid_extensions


def validate_image_paths(image_paths: List[str]) -> List[str]:
    """
    验证图片路径列表，返回有效的路径
    
    Args:
        image_paths: 图片路径列表
        
    Returns:
        有效的图片路径列表
    """
    valid_paths = []
    for path in image_paths:
        if validate_image_path(path):
            valid_paths.append(path)
    return valid_paths


def validate_video_path(video_path: str) -> bool:
    """
    验证视频路径是否有效
    
    Args:
        video_path: 视频路径
        
    Returns:
        是否有效
    """
    if not video_path or not isinstance(video_path, str):
        return False
    
    # 检查是否为摄像头ID
    if video_path.isdigit():
        return True
    
    # 检查RTSP流
    if video_path.startswith('rtsp://'):
        return True
    
    # 检查文件路径
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        return False
    
    # 检查视频文件扩展名
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    return path.suffix.lower() in valid_extensions


def validate_model_path(model_path: str) -> bool:
    """
    验证模型路径是否有效
    
    Args:
        model_path: 模型路径
        
    Returns:
        是否有效
    """
    if not model_path or not isinstance(model_path, str):
        return False
    
    path = Path(model_path)
    if not path.exists() or not path.is_file():
        return False
    
    # 检查是否为ONNX模型文件
    return path.suffix.lower() == '.onnx'


def validate_confidence_threshold(threshold: float) -> bool:
    """
    验证置信度阈值是否有效
    
    Args:
        threshold: 置信度阈值
        
    Returns:
        是否有效
    """
    return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除非法字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        清理后的文件名
    """
    import re
    # 移除或替换非法字符
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除连续的下划线
    sanitized = re.sub(r'_+', '_', sanitized)
    # 移除开头和结尾的点和空格
    sanitized = sanitized.strip('. ')
    return sanitized or 'unnamed'


def is_base64_image(data: str) -> bool:
    """
    检查字符串是否为有效的base64图片数据
    
    Args:
        data: 输入数据字符串
        
    Returns:
        是否为有效的base64图片
    """
    if not isinstance(data, str):
        return False
    
    # 检查是否为data URL格式
    if data.startswith('data:image/'):
        return True
    
    # 尝试解码base64
    try:
        # 移除可能的data URL前缀
        if ',' in data:
            data = data.split(',', 1)[1]
        
        # 检查base64格式
        decoded = base64.b64decode(data, validate=True)
        
        # 检查图片文件头
        image_signatures = [
            b'\xFF\xD8\xFF',  # JPEG
            b'\x89PNG',       # PNG
            b'GIF87a',        # GIF87a
            b'GIF89a',        # GIF89a
            b'BM',            # BMP
            b'RIFF',          # WebP (RIFF header)
        ]
        
        return any(decoded.startswith(sig) for sig in image_signatures)
    except Exception:
        return False


def decode_base64_image(data: str) -> Tuple[str, str]:
    """
    解码base64图片数据并保存为临时文件
    
    Args:
        data: base64图片数据
        
    Returns:
        临时文件路径和图片格式
        
    Raises:
        ValueError: 如果数据无效
    """
    if not is_base64_image(data):
        raise ValueError("无效的base64图片数据")
    
    try:
        # 提取图片格式和数据
        if data.startswith('data:image/'):
            header, base64_data = data.split(',', 1)
            # 从header中提取格式 data:image/jpeg;base64,
            format_match = re.search(r'data:image/([^;]+)', header)
            image_format = format_match.group(1) if format_match else 'jpg'
        else:
            base64_data = data
            image_format = 'jpg'  # 默认格式
        
        # 检查数据大小（大约估算解码后的大小）
        estimated_size = len(base64_data) * 3 // 4  # base64解码后大小约为原来的3/4
        max_size = 50 * 1024 * 1024  # 50MB限制
        
        if estimated_size > max_size:
            raise ValueError(f"图片太大: 估算大小 {estimated_size // (1024*1024)}MB，超过50MB限制")
        
        # 解码数据
        decoded_data = base64.b64decode(base64_data)
        
        # 检查实际解码大小
        if len(decoded_data) > max_size:
            raise ValueError(f"图片太大: {len(decoded_data) // (1024*1024)}MB，超过50MB限制")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=f'.{image_format}', delete=False) as temp_file:
            temp_file.write(decoded_data)
            temp_path = temp_file.name
        
        return temp_path, image_format
        
    except Exception as e:
        raise ValueError(f"解码base64图片失败: {e}")


def validate_image_input(image_input: str) -> Tuple[str, bool]:
    """
    验证图片输入，支持文件路径和base64数据
    
    Args:
        image_input: 图片路径或base64数据
        
    Returns:
        (有效的文件路径, 是否为临时文件)
        
    Raises:
        ValueError: 如果输入无效
    """
    if not image_input or not isinstance(image_input, str):
        raise ValueError("图片输入不能为空")
    
    # 处理特殊的Claude Desktop输入值
    if image_input.lower() in ['image', 'picture', 'photo', 'img']:
        # 这种情况下，我们需要一个默认的测试图片或提示用户提供具体路径
        raise ValueError(
            f"检测到通用图片引用 '{image_input}'。"
            f"请提供具体的图片文件路径，如: /path/to/image.jpg，"
            f"或直接将图片拖拽到聊天框中。"
        )
    
    # 检查是否为base64数据
    if is_base64_image(image_input):
        temp_path, _ = decode_base64_image(image_input)
        return temp_path, True
    
    # 检查是否为有效的文件路径
    if validate_image_path(image_input):
        return image_input, False
    
    # 提供更详细的错误信息
    if len(image_input) < 50:  # 短字符串，可能是错误的路径
        raise ValueError(
            f"无效的图片路径: '{image_input}'。"
            f"请确保路径正确且文件存在，"
            f"支持格式: .jpg, .jpeg, .png, .bmp, .tiff, .webp"
        )
    else:  # 长字符串，可能是格式错误的base64
        raise ValueError(
            f"无效的图片数据。"
            f"如果这是base64数据，请确保格式正确（以data:image/开头或纯base64字符串）。"
            f"如果这是文件路径，请检查路径是否正确。"
        )