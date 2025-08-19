"""
配置模型定义
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ModelConfig(BaseModel):
    """模型配置"""
    model_path: str = Field(description="模型文件路径")
    input_shape: tuple[int, int] = Field(default=(640, 640), description="输入图像尺寸")
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="NMS IoU阈值")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f'模型文件不存在: {v}')
        return v


class DetectionConfig(BaseModel):
    """检测配置"""
    detection_model: ModelConfig = Field(description="主检测模型配置")
    color_layer_model: ModelConfig = Field(description="颜色层数分类模型配置")
    ocr_model: ModelConfig = Field(description="OCR模型配置")
    class_names: List[str] = Field(description="类别名称列表")
    colors: Dict[int, List[int]] = Field(description="可视化颜色映射")
    roi_top_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="ROI上边比例")
    plate_conf_threshold: Optional[float] = Field(None, description="车牌特定置信度阈值")


class ServerConfig(BaseModel):
    """服务器配置"""
    name: str = Field(default="Vehicle Detection Server", description="服务器名称")
    version: str = Field(default="1.0.0", description="服务器版本")
    description: str = Field(default="车辆和车牌检测MCP服务器", description="服务器描述")
    max_concurrent_sessions: int = Field(default=10, ge=1, description="最大并发会话数")
    session_timeout: int = Field(default=3600, ge=60, description="会话超时时间（秒）")
    cache_size: int = Field(default=100, ge=1, description="结果缓存大小")
    log_level: str = Field(default="INFO", description="日志级别")
    
    
class ProcessingConfig(BaseModel):
    """处理配置"""
    frame_skip: int = Field(default=0, ge=0, description="视频帧跳跃间隔")
    batch_size: int = Field(default=1, ge=1, description="批处理大小")
    max_image_size: int = Field(default=50 * 1024 * 1024, ge=1024*1024, description="最大图像文件大小（字节）")
    max_image_dimension: int = Field(default=4096, ge=256, description="最大图像尺寸（像素）")
    supported_formats: List[str] = Field(
        default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        description="支持的图像格式"
    )
    max_video_duration: int = Field(default=3600, ge=1, description="最大视频时长（秒）")
    enable_image_optimization: bool = Field(default=True, description="启用图像预处理优化")