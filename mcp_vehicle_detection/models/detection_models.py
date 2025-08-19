"""
检测结果数据模型定义
使用Pydantic进行数据验证和序列化
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
from datetime import datetime


class DetectionType(str, Enum):
    """检测对象类型枚举"""
    VEHICLE = "vehicle"
    PLATE = "plate"
    UNKNOWN = "unknown"


class PlateColor(str, Enum):
    """车牌颜色枚举"""
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


class PlateLayer(str, Enum):
    """车牌层数枚举"""
    SINGLE = "single"
    DOUBLE = "double"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """边界框模型"""
    x1: float = Field(description="左上角X坐标")
    y1: float = Field(description="左上角Y坐标")
    x2: float = Field(description="右下角X坐标") 
    y2: float = Field(description="右下角Y坐标")
    
    @validator('x1', 'x2', 'y1', 'y2')
    def check_coordinates(cls, v):
        if v < 0:
            raise ValueError('坐标值不能为负数')
        return v
    
    @property
    def width(self) -> float:
        """计算宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """计算高度"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """计算面积"""
        return self.width * self.height


class PlateInfo(BaseModel):
    """车牌信息模型"""
    text: str = Field(description="车牌文字识别结果")
    color: PlateColor = Field(description="车牌颜色")
    layer: PlateLayer = Field(description="车牌层数")
    confidence: float = Field(ge=0.0, le=1.0, description="OCR识别置信度")
    should_display_ocr: bool = Field(default=True, description="是否应显示OCR结果")


class Detection(BaseModel):
    """单个检测结果模型"""
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="检测结果唯一标识")
    type: DetectionType = Field(description="检测对象类型")
    bbox: BoundingBox = Field(description="边界框")
    confidence: float = Field(ge=0.0, le=1.0, description="检测置信度")
    class_id: int = Field(ge=0, description="类别ID")
    plate_info: Optional[PlateInfo] = Field(None, description="车牌详细信息（仅车牌类型有效）")
    
    @validator('plate_info', always=True)
    def validate_plate_info(cls, v, values):
        if values.get('type') == DetectionType.PLATE and v is None:
            raise ValueError('车牌类型检测结果必须包含车牌信息')
        elif values.get('type') != DetectionType.PLATE and v is not None:
            raise ValueError('非车牌类型检测结果不应包含车牌信息')
        return v


class ImageInfo(BaseModel):
    """图像信息模型"""
    width: int = Field(gt=0, description="图像宽度")
    height: int = Field(gt=0, description="图像高度")
    channels: int = Field(ge=1, le=4, description="图像通道数")
    file_path: Optional[str] = Field(None, description="图像文件路径")


class DetectionResult(BaseModel):
    """完整检测结果模型"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="检测会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="检测时间戳")
    image_info: ImageInfo = Field(description="图像信息")
    detections: List[Detection] = Field(default=[], description="检测结果列表")
    processing_time: float = Field(ge=0.0, description="处理耗时（秒）")
    model_info: Dict[str, str] = Field(description="模型信息")
    
    @property
    def vehicle_count(self) -> int:
        """车辆数量"""
        return len([d for d in self.detections if d.type == DetectionType.VEHICLE])
    
    @property
    def plate_count(self) -> int:
        """车牌数量"""
        return len([d for d in self.detections if d.type == DetectionType.PLATE])
    
    @property
    def plate_texts(self) -> List[str]:
        """所有车牌文字"""
        return [d.plate_info.text for d in self.detections 
                if d.type == DetectionType.PLATE and d.plate_info]


class BatchDetectionResult(BaseModel):
    """批量检测结果模型"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="批次ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="批次开始时间")
    results: List[DetectionResult] = Field(default=[], description="检测结果列表")
    total_images: int = Field(ge=0, description="总图片数量")
    processed_images: int = Field(ge=0, description="已处理图片数量")
    failed_images: int = Field(ge=0, description="处理失败图片数量")
    total_processing_time: float = Field(ge=0.0, description="总处理时间（秒）")
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_images == 0:
            return 0.0
        return self.processed_images / self.total_images
    
    @property
    def average_processing_time(self) -> float:
        """平均处理时间"""
        if self.processed_images == 0:
            return 0.0
        return self.total_processing_time / self.processed_images


class DetectionStatistics(BaseModel):
    """检测统计信息模型"""
    total_detections: int = Field(ge=0, description="总检测次数")
    total_vehicles: int = Field(ge=0, description="总车辆数量")
    total_plates: int = Field(ge=0, description="总车牌数量")
    average_processing_time: float = Field(ge=0.0, description="平均处理时间")
    most_common_plate_color: Optional[PlateColor] = Field(None, description="最常见车牌颜色")
    detection_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="检测准确率")


class VideoAnalysisResult(BaseModel):
    """视频分析结果模型"""
    video_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="视频分析ID")
    video_path: str = Field(description="视频文件路径")
    total_frames: int = Field(ge=0, description="总帧数")
    processed_frames: int = Field(ge=0, description="已处理帧数")
    frame_results: Dict[int, DetectionResult] = Field(default={}, description="帧检测结果")
    start_time: datetime = Field(default_factory=datetime.now, description="分析开始时间")
    end_time: Optional[datetime] = Field(None, description="分析结束时间")
    
    @property
    def processing_progress(self) -> float:
        """处理进度"""
        if self.total_frames == 0:
            return 0.0
        return self.processed_frames / self.total_frames
    
    @property
    def unique_plates(self) -> List[str]:
        """视频中出现的所有唯一车牌"""
        plates = set()
        for result in self.frame_results.values():
            plates.update(result.plate_texts)
        return list(plates)