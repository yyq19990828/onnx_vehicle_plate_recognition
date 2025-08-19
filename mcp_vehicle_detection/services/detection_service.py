"""
检测服务封装
将现有的检测功能封装为可复用的服务类
"""

import sys
import os
import cv2
import numpy as np
import yaml
import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.detection_models import (
    DetectionResult, Detection, BoundingBox, PlateInfo, ImageInfo,
    DetectionType, PlateColor, PlateLayer, BatchDetectionResult
)
from models.config_models import DetectionConfig, ProcessingConfig


class VehicleDetectionService:
    """车辆检测服务"""
    
    def __init__(self, config: DetectionConfig):
        """
        初始化检测服务
        
        Args:
            config: 检测配置
        """
        self.config = config
        self.detector = None
        self.color_layer_classifier = None  
        self.ocr_model = None
        self.character = None
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化所有模型"""
        try:
            # 添加父目录到路径以访问infer_onnx模块
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # 导入检测模块
            from infer_onnx import RTDETROnnx, ColorLayerONNX, OCRONNX
            
            # 初始化主检测模型 - RT-DETR不需要iou_thres参数
            self.detector = RTDETROnnx(
                self.config.detection_model.model_path,
                input_shape=tuple(self.config.detection_model.input_shape),
                conf_thres=self.config.detection_model.conf_threshold
            )
            
            # 初始化颜色层数分类模型
            self.color_layer_classifier = ColorLayerONNX(
                self.config.color_layer_model.model_path
            )
            
            # 初始化OCR模型
            self.ocr_model = OCRONNX(
                self.config.ocr_model.model_path
            )
            
            # 加载OCR字符集
            ocr_dict_path = Path(self.config.ocr_model.model_path).parent / "ocr_dict.yaml"
            with open(ocr_dict_path, "r", encoding="utf-8") as f:
                dict_yaml = yaml.safe_load(f)
                self.character = ["blank"] + dict_yaml["ocr_dict"] + [" "]
            
            logging.info("所有模型初始化成功")
            
        except Exception as e:
            logging.error(f"模型初始化失败: {e}")
            raise
    
    def detect_single_image(self, image_path: str, **kwargs) -> DetectionResult:
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            **kwargs: 额外参数
            
        Returns:
            检测结果
        """
        start_time = time.time()
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 获取图片信息
        h, w, c = image.shape
        image_info = ImageInfo(
            width=w,
            height=h, 
            channels=c,
            file_path=image_path
        )
        
        # 执行检测
        detections, _ = self.detector(image)
        
        # 处理检测结果
        detection_objects = []
        if detections and len(detections[0]) > 0:
            detection_objects = self._process_detections(
                image, detections[0], image_info
            )
        
        processing_time = time.time() - start_time
        
        # 构建结果
        result = DetectionResult(
            image_info=image_info,
            detections=detection_objects,
            processing_time=processing_time,
            model_info={
                "detection_model": self.config.detection_model.model_path,
                "ocr_model": self.config.ocr_model.model_path,
                "color_layer_model": self.config.color_layer_model.model_path
            }
        )
        
        return result
    
    def detect_batch_images(self, image_paths: List[str], **kwargs) -> BatchDetectionResult:
        """
        批量检测图片
        
        Args:
            image_paths: 图片路径列表
            **kwargs: 额外参数
            
        Returns:
            批量检测结果
        """
        start_time = time.time()
        results = []
        failed_count = 0
        
        for image_path in image_paths:
            try:
                result = self.detect_single_image(image_path, **kwargs)
                results.append(result)
            except Exception as e:
                logging.error(f"处理图片失败 {image_path}: {e}")
                failed_count += 1
        
        total_time = time.time() - start_time
        
        batch_result = BatchDetectionResult(
            results=results,
            total_images=len(image_paths),
            processed_images=len(results),
            failed_images=failed_count,
            total_processing_time=total_time
        )
        
        return batch_result
    
    def _process_detections(self, image: np.ndarray, detections: np.ndarray, 
                          image_info: ImageInfo) -> List[Detection]:
        """
        处理检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果数组
            image_info: 图像信息
            
        Returns:
            检测对象列表
        """
        detection_objects = []
        h_img, w_img = image_info.height, image_info.width
        roi_top_pixel = int(h_img * self.config.roi_top_ratio)
        
        # 裁剪检测框到图像边界内
        clipped_detections = detections.copy()
        clipped_detections[:, 0] = np.clip(clipped_detections[:, 0], 0, w_img)
        clipped_detections[:, 1] = np.clip(clipped_detections[:, 1], 0, h_img)
        clipped_detections[:, 2] = np.clip(clipped_detections[:, 2], 0, w_img)
        clipped_detections[:, 3] = np.clip(clipped_detections[:, 3], 0, h_img)
        
        plate_conf_thres = (self.config.plate_conf_threshold if self.config.plate_conf_threshold 
                           else self.config.detection_model.conf_threshold)
        
        for detection_idx, (*xyxy, conf, cls) in enumerate(clipped_detections):
            class_name = (self.config.class_names[int(cls)] if int(cls) < len(self.config.class_names) 
                         else "unknown")
            
            # 应用车牌特定置信度阈值
            if class_name == 'plate' and conf < plate_conf_thres:
                continue
            
            # 创建边界框
            bbox = BoundingBox(x1=float(xyxy[0]), y1=float(xyxy[1]),
                             x2=float(xyxy[2]), y2=float(xyxy[3]))
            
            # 确定检测类型
            detection_type = DetectionType.PLATE if class_name == 'plate' else DetectionType.VEHICLE
            
            # 处理车牌信息
            plate_info = None
            if detection_type == DetectionType.PLATE:
                plate_info = self._process_plate(image, bbox, roi_top_pixel)
            
            # 创建检测对象
            detection_obj = Detection(
                type=detection_type,
                bbox=bbox,
                confidence=float(conf),
                class_id=int(cls),
                plate_info=plate_info
            )
            
            detection_objects.append(detection_obj)
        
        return detection_objects
    
    def _process_plate(self, image: np.ndarray, bbox: BoundingBox, 
                      roi_top_pixel: int) -> PlateInfo:
        """
        处理车牌检测结果
        
        Args:
            image: 原始图像
            bbox: 车牌边界框
            roi_top_pixel: ROI顶部像素位置
            
        Returns:
            车牌信息
        """
        try:
            # 导入图像处理函数
            from utils import process_plate_image, image_pretreatment, resize_norm_img, decode
            
            h_img, w_img = image.shape[:2]
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            w, h = x2 - x1, y2 - y1
            
            # 扩展车牌区域
            exp_x1 = int(max(0, x1 - w * 0.1))
            exp_y1 = int(max(0, y1 - h * 0.1))
            exp_x2 = int(min(w_img, x2 + w * 0.1))
            exp_y2 = int(min(h_img, y2 + h * 0.1))
            plate_img = image[exp_y1:exp_y2, exp_x1:exp_x2]
            
            if plate_img.size == 0:
                return PlateInfo(
                    text="", color=PlateColor.UNKNOWN, layer=PlateLayer.UNKNOWN,
                    confidence=0.0, should_display_ocr=False
                )
            
            # 颜色和层数分类
            img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            color_input = image_pretreatment(img_rgb)
            preds_color, preds_layer = self.color_layer_classifier.infer(color_input)
            color_index = int(np.argmax(preds_color))
            layer_index = int(np.argmax(preds_layer))
            
            # 加载颜色层数映射
            color_layer_path = Path(self.config.color_layer_model.model_path).parent / "plate_color_layer.yaml"
            with open(color_layer_path, "r", encoding="utf-8") as f:
                color_layer_yaml = yaml.safe_load(f)
            
            color_dict = color_layer_yaml["color_dict"]
            layer_dict = color_layer_yaml["layer_dict"]
            color_str = color_dict.get(color_index, "unknown")
            layer_str = layer_dict.get(layer_index, "unknown")
            
            # OCR识别
            is_double = (layer_str == "double")
            processed_plate = process_plate_image(plate_img, is_double_layer=is_double)
            ocr_input = resize_norm_img(processed_plate)
            ocr_out = self.ocr_model.infer(ocr_input)
            preds_idx = np.asarray(ocr_out[0]).argmax(axis=2)
            preds_prob = np.asarray(ocr_out[0]).max(axis=2)
            ocr_result = decode(self.character, preds_idx, preds_prob, is_remove_duplicate=True)
            plate_text = ocr_result[0][0] if ocr_result else ""
            
            # 计算OCR置信度
            ocr_confidence = float(np.mean(preds_prob)) if len(preds_prob) > 0 else 0.0
            
            # 判断是否显示OCR结果
            should_display_ocr = (y1 >= roi_top_pixel) and (w > 50)
            
            return PlateInfo(
                text=plate_text,
                color=PlateColor(color_str) if color_str in PlateColor.__members__.values() else PlateColor.UNKNOWN,
                layer=PlateLayer(layer_str) if layer_str in PlateLayer.__members__.values() else PlateLayer.UNKNOWN,
                confidence=ocr_confidence,
                should_display_ocr=should_display_ocr
            )
            
        except Exception as e:
            logging.error(f"车牌处理失败: {e}")
            return PlateInfo(
                text="", color=PlateColor.UNKNOWN, layer=PlateLayer.UNKNOWN,
                confidence=0.0, should_display_ocr=False
            )