"""
图片处理工具模块
借鉴DINO-X-MCP的图片处理方案，支持多种图片输入格式
"""

import os
import base64
import tempfile
import requests
from typing import Tuple, Optional, Union
from pathlib import Path
from urllib.parse import urlparse
import io
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图片处理器，支持多种输入格式"""
    
    def __init__(self, max_image_size: int = 50 * 1024 * 1024, max_dimension: int = 4096):
        """
        初始化图片处理器
        
        Args:
            max_image_size: 最大图片大小限制（字节）
            max_dimension: 最大图片尺寸限制（像素），用于内存优化
        """
        self.max_image_size = max_image_size
        self.max_dimension = max_dimension
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.temp_files = []  # 跟踪临时文件用于清理
    
    def process_image_uri(self, image_uri: str) -> Tuple[str, bool]:
        """
        处理图片URI，支持多种格式
        借鉴DINO-X-MCP的processImageUri函数设计
        
        Args:
            image_uri: 图片URI，支持以下格式：
                     - file:// URI (本地文件)
                     - https:// URI (网络图片) 
                     - data:image/ URI (base64图片)
                     - 普通文件路径
                     - 纯base64字符串
        
        Returns:
            (本地文件路径, 是否为临时文件)
            
        Raises:
            ValueError: 如果URI格式无效或处理失败
        """
        if not image_uri or not isinstance(image_uri, str):
            raise ValueError("图片URI不能为空")
        
        # 1. 处理 file:// URI
        if image_uri.startswith("file://"):
            return self._process_file_uri(image_uri)
        
        # 2. 处理 https:// URI  
        elif image_uri.startswith("https://") or image_uri.startswith("http://"):
            return self._process_http_uri(image_uri)
        
        # 3. 处理 data:image/ URI (base64)
        elif image_uri.startswith("data:image/"):
            return self._process_data_uri(image_uri)
        
        # 4. 处理纯base64字符串
        elif self._is_base64_string(image_uri):
            return self._process_base64_string(image_uri)
        
        # 5. 处理普通文件路径
        elif self._is_valid_file_path(image_uri):
            return image_uri, False
        
        else:
            raise ValueError(f"不支持的图片URI格式: {image_uri[:100]}...")
    
    def _process_file_uri(self, file_uri: str) -> Tuple[str, bool]:
        """处理 file:// URI"""
        try:
            from urllib.parse import unquote
            from urllib.request import url2pathname
            
            # 移除 file:// 前缀并解码
            path = unquote(file_uri[7:])  # 移除 "file://"
            
            # 处理Windows路径
            if os.name == 'nt' and path.startswith('/'):
                path = path[1:]
            
            if not os.path.exists(path):
                raise ValueError(f"文件不存在: {path}")
            
            if not self._is_valid_image_file(path):
                raise ValueError(f"不支持的图片格式: {path}")
            
            return path, False
            
        except Exception as e:
            raise ValueError(f"处理file:// URI失败: {e}")
    
    def _process_http_uri(self, http_uri: str) -> Tuple[str, bool]:
        """处理 https:// 或 http:// URI"""
        try:
            # 验证URL格式
            parsed = urlparse(http_uri)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("无效的HTTP URL格式")
            
            # 下载图片
            response = requests.get(http_uri, timeout=30, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"可能不是图片文件，Content-Type: {content_type}")
            
            # 检查文件大小
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_image_size:
                raise ValueError(f"图片太大: {int(content_length)} 字节")
            
            # 确定文件扩展名
            ext = self._get_extension_from_url_or_content_type(http_uri, content_type)
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded_size += len(chunk)
                    if downloaded_size > self.max_image_size:
                        raise ValueError(f"图片太大: 超过 {self.max_image_size} 字节")
                    temp_file.write(chunk)
                
                temp_path = temp_file.name
                self.temp_files.append(temp_path)
                return temp_path, True
            
        except Exception as e:
            raise ValueError(f"下载网络图片失败: {e}")
    
    def _process_data_uri(self, data_uri: str) -> Tuple[str, bool]:
        """处理 data:image/ URI"""
        try:
            # 解析 data URI: data:image/jpeg;base64,/9j/4AAQ...
            if ',' not in data_uri:
                raise ValueError("无效的data URI格式")
            
            header, base64_data = data_uri.split(',', 1)
            
            # 提取图片格式
            import re
            format_match = re.search(r'data:image/([^;]+)', header)
            if not format_match:
                raise ValueError("无法识别data URI中的图片格式")
            
            image_format = format_match.group(1).lower()
            if image_format not in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff']:
                raise ValueError(f"不支持的图片格式: {image_format}")
            
            return self._decode_base64_to_file(base64_data, image_format)
            
        except Exception as e:
            raise ValueError(f"处理data URI失败: {e}")
    
    def _process_base64_string(self, base64_string: str) -> Tuple[str, bool]:
        """处理纯base64字符串"""
        return self._decode_base64_to_file(base64_string, 'jpg')
    
    def _decode_base64_to_file(self, base64_data: str, format_ext: str) -> Tuple[str, bool]:
        """解码base64数据到文件"""
        try:
            # 检查数据大小
            estimated_size = len(base64_data) * 3 // 4
            if estimated_size > self.max_image_size:
                raise ValueError(f"图片太大: 估算大小 {estimated_size // (1024*1024)}MB")
            
            # 解码
            decoded_data = base64.b64decode(base64_data, validate=True)
            
            # 验证实际大小
            if len(decoded_data) > self.max_image_size:
                raise ValueError(f"图片太大: {len(decoded_data) // (1024*1024)}MB")
            
            # 验证图片格式
            if not self._verify_image_signature(decoded_data):
                raise ValueError("不是有效的图片数据")
            
            # 保存到临时文件
            ext = f'.{format_ext}' if not format_ext.startswith('.') else format_ext
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(decoded_data)
                temp_path = temp_file.name
                self.temp_files.append(temp_path)
                return temp_path, True
                
        except Exception as e:
            raise ValueError(f"解码base64数据失败: {e}")
    
    def _is_base64_string(self, s: str) -> bool:
        """检查字符串是否为base64格式"""
        if len(s) < 100:  # 太短不太可能是图片的base64
            return False
        
        try:
            import re
            # 检查base64字符集
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
                return False
            
            decoded = base64.b64decode(s, validate=True)
            return self._verify_image_signature(decoded)
        except Exception:
            return False
    
    def _verify_image_signature(self, data: bytes) -> bool:
        """验证图片文件签名"""
        if len(data) < 8:
            return False
        
        # 图片文件头签名
        signatures = [
            (b'\xFF\xD8\xFF', 'JPEG'),
            (b'\x89PNG\r\n\x1A\n', 'PNG'),
            (b'GIF87a', 'GIF87a'),
            (b'GIF89a', 'GIF89a'),
            (b'BM', 'BMP'),
            (b'RIFF', 'WebP'),  # WebP uses RIFF container
            (b'MM\x00\x2A', 'TIFF BE'),  # TIFF big-endian
            (b'II\x2A\x00', 'TIFF LE'),  # TIFF little-endian
        ]
        
        for sig, fmt in signatures:
            if data.startswith(sig):
                return True
        
        return False
    
    def _is_valid_file_path(self, path: str) -> bool:
        """检查是否为有效的图片文件路径"""
        try:
            p = Path(path)
            return (p.exists() and 
                    p.is_file() and 
                    p.suffix.lower() in self.supported_formats)
        except Exception:
            return False
    
    def _is_valid_image_file(self, path: str) -> bool:
        """检查文件是否为有效的图片文件"""
        return self._is_valid_file_path(path)
    
    def _get_extension_from_url_or_content_type(self, url: str, content_type: str) -> str:
        """从URL或Content-Type获取文件扩展名"""
        # 首先尝试从URL获取
        parsed = urlparse(url)
        path = Path(parsed.path)
        if path.suffix.lower() in self.supported_formats:
            return path.suffix
        
        # 从Content-Type获取
        content_type_map = {
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg', 
            'image/png': '.png',
            'image/bmp': '.bmp',
            'image/webp': '.webp',
            'image/tiff': '.tiff'
        }
        
        return content_type_map.get(content_type.lower(), '.jpg')
    
    def preprocess_for_model(self, image_path: str) -> str:
        """
        为模型推理预处理图片，包括内存优化
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            预处理后的图片文件路径（可能是新的临时文件）
        """
        try:
            import cv2
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            h, w = image.shape[:2]
            
            # 如果图片尺寸过大，进行预缩放以节省内存和提升性能
            if max(h, w) > self.max_dimension:
                logger.info(f"图片尺寸过大 ({w}x{h})，进行预缩放优化")
                
                # 计算预缩放比例
                scale = self.max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # 根据缩放比例选择合适的插值方法
                if scale < 1.0:
                    # 缩小时使用INTER_AREA获得更好效果
                    interpolation = cv2.INTER_AREA
                else:
                    # 放大时使用INTER_CUBIC（虽然这里不太可能）
                    interpolation = cv2.INTER_CUBIC
                
                # 进行预缩放
                resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
                
                # 保存预缩放后的图片到临时文件
                import tempfile
                ext = os.path.splitext(image_path)[1] or '.jpg'
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, resized_image)
                    temp_path = temp_file.name
                    self.temp_files.append(temp_path)
                
                logger.info(f"预缩放完成: {w}x{h} -> {new_w}x{new_h}, 保存到: {temp_path}")
                return temp_path
            else:
                # 尺寸合理，直接返回原路径
                return image_path
                
        except Exception as e:
            logger.warning(f"图片预处理失败，使用原图片: {e}")
            return image_path
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError as e:
                logger.warning(f"清理临时文件失败: {temp_file}, {e}")
        
        self.temp_files.clear()
    
    def __del__(self):
        """析构时清理临时文件"""
        self.cleanup_temp_files()


# 全局图片处理器实例
_global_processor = None


def get_image_processor() -> ImageProcessor:
    """获取全局图片处理器实例"""
    global _global_processor
    if _global_processor is None:
        _global_processor = ImageProcessor()
    return _global_processor


def process_image_input(image_input: str) -> Tuple[str, bool]:
    """
    便捷函数：处理图片输入
    
    Args:
        image_input: 图片输入（路径、URL、base64等）
        
    Returns:
        (本地文件路径, 是否为临时文件)
    """
    processor = get_image_processor()
    return processor.process_image_uri(image_input)


def cleanup_temp_files():
    """便捷函数：清理临时文件"""
    global _global_processor
    if _global_processor:
        _global_processor.cleanup_temp_files()