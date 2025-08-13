"""
输出转换函数模块

此模块提供将不同模型的输出格式转换为YOLO标准格式的转换函数。
YOLO标准格式: [x1, y1, x2, y2, confidence, class_id]
"""

import numpy as np
from typing import List, Tuple


def rfdetr_to_yolo_transform(detections: List[np.ndarray], original_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    将RF-DETR模型的输出转换为YOLO格式
    
    RF-DETR已经输出标准的YOLO格式，所以这里直接返回
    
    Args:
        detections (List[np.ndarray]): RF-DETR的检测结果
        original_shape (Tuple[int, int]): 原始图像尺寸 (height, width)
        
    Returns:
        List[np.ndarray]: YOLO格式的检测结果
    """
    # RF-DETR类已经处理了格式转换，直接返回
    return detections


def custom_model_transform_example(detections: List[np.ndarray], original_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    自定义模型输出转换示例
    
    这是一个示例函数，展示如何将自定义模型的输出转换为YOLO格式
    用户可以根据自己的模型输出格式修改这个函数
    
    Args:
        detections (List[np.ndarray]): 模型的原始检测结果
        original_shape (Tuple[int, int]): 原始图像尺寸 (height, width)
        
    Returns:
        List[np.ndarray]: YOLO格式的检测结果 [x1, y1, x2, y2, confidence, class_id]
    """
    if not detections or len(detections[0]) == 0:
        return detections
    
    # 示例：假设输入格式为 [center_x, center_y, width, height, confidence, class_id]
    # 需要转换为 [x1, y1, x2, y2, confidence, class_id]
    
    converted_detections = []
    
    for detection_batch in detections:
        if len(detection_batch) == 0:
            converted_detections.append(detection_batch)
            continue
        
        converted_batch = detection_batch.copy()
        
        # 提取中心坐标和宽高
        cx = detection_batch[:, 0]
        cy = detection_batch[:, 1] 
        w = detection_batch[:, 2]
        h = detection_batch[:, 3]
        
        # 转换为左上角和右下角坐标
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 更新坐标
        converted_batch[:, 0] = x1
        converted_batch[:, 1] = y1
        converted_batch[:, 2] = x2
        converted_batch[:, 3] = y2
        # confidence和class_id保持不变 (columns 4 and 5)
        
        converted_detections.append(converted_batch)
    
    return converted_detections


def yolov5_to_yolo_transform(detections: List[np.ndarray], original_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    将YOLOv5模型的输出转换为YOLO格式
    
    YOLOv5通常输出格式已经是 [x1, y1, x2, y2, confidence, class_id]，所以直接返回
    
    Args:
        detections (List[np.ndarray]): YOLOv5的检测结果
        original_shape (Tuple[int, int]): 原始图像尺寸 (height, width)
        
    Returns:
        List[np.ndarray]: YOLO格式的检测结果
    """
    return detections


def normalize_bbox_transform(detections: List[np.ndarray], original_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    将归一化的边界框坐标转换为绝对坐标的YOLO格式
    
    Args:
        detections (List[np.ndarray]): 检测结果，坐标为归一化格式 [0, 1]
        original_shape (Tuple[int, int]): 原始图像尺寸 (height, width)
        
    Returns:
        List[np.ndarray]: YOLO格式的检测结果，坐标为绝对像素值
    """
    if not detections or len(detections[0]) == 0:
        return detections
    
    h, w = original_shape
    converted_detections = []
    
    for detection_batch in detections:
        if len(detection_batch) == 0:
            converted_detections.append(detection_batch)
            continue
        
        converted_batch = detection_batch.copy()
        
        # 将归一化坐标转换为绝对坐标
        converted_batch[:, 0] *= w  # x1
        converted_batch[:, 1] *= h  # y1  
        converted_batch[:, 2] *= w  # x2
        converted_batch[:, 3] *= h  # y2
        # confidence和class_id保持不变
        
        converted_detections.append(converted_batch)
    
    return converted_detections


def cxcywh_to_xyxy_transform(detections: List[np.ndarray], original_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    将中心坐标+宽高格式转换为YOLO的xyxy格式
    
    Args:
        detections (List[np.ndarray]): 输入格式 [center_x, center_y, width, height, confidence, class_id]
        original_shape (Tuple[int, int]): 原始图像尺寸 (height, width)
        
    Returns:
        List[np.ndarray]: YOLO格式 [x1, y1, x2, y2, confidence, class_id]
    """
    if not detections or len(detections[0]) == 0:
        return detections
    
    converted_detections = []
    
    for detection_batch in detections:
        if len(detection_batch) == 0:
            converted_detections.append(detection_batch)
            continue
        
        converted_batch = detection_batch.copy()
        
        # 提取中心坐标和宽高
        cx = detection_batch[:, 0]
        cy = detection_batch[:, 1]
        w = detection_batch[:, 2] 
        h = detection_batch[:, 3]
        
        # 转换为xyxy格式
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 更新坐标
        converted_batch[:, 0] = x1
        converted_batch[:, 1] = y1
        converted_batch[:, 2] = x2
        converted_batch[:, 3] = y2
        
        converted_detections.append(converted_batch)
    
    return converted_detections


# 预定义的转换函数字典，方便调用
TRANSFORM_FUNCTIONS = {
    'rfdetr': rfdetr_to_yolo_transform,
    'yolov5': yolov5_to_yolo_transform,
    'normalize_bbox': normalize_bbox_transform,
    'cxcywh_to_xyxy': cxcywh_to_xyxy_transform,
    'custom_example': custom_model_transform_example,
}


def get_transform_function(transform_name: str):
    """
    根据名称获取转换函数
    
    Args:
        transform_name (str): 转换函数名称
        
    Returns:
        Callable: 转换函数
        
    Raises:
        ValueError: 如果转换函数不存在
    """
    if transform_name not in TRANSFORM_FUNCTIONS:
        available = ', '.join(TRANSFORM_FUNCTIONS.keys())
        raise ValueError(f"未知的转换函数: {transform_name}。可用的转换函数: {available}")
    
    return TRANSFORM_FUNCTIONS[transform_name]