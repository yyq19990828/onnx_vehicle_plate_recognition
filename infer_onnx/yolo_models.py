"""
统一的YOLO系列ONNX模型推理类

包含:
- YoloOnnx: 传统YOLO模型基类 (原DetONNX)
- RTDETROnnx: RT-DETR模型类，继承自YoloOnnx (原YoloRTDETROnnx)
- RFDETROnnx: RF-DETR模型类

统一API设计，减少代码冗余，提高可维护性
"""

import numpy as np
import logging
import time
import cv2
import yaml
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Callable
from pathlib import Path

from .utils import get_model_info
from utils.image_processing import preprocess_image, preprocess_image_ultralytics
from utils.nms import non_max_suppression
from utils.detection_metrics import evaluate_detection, print_metrics


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.
    
    Source: ultralytics/utils/ops.py::xywh2xyxy
    原函数路径: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py
    
    Args:
        x (np.ndarray): Input bounding box coordinates in (x, y, width, height) format.
        
    Returns:
        np.ndarray: Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.
    
    Source: ultralytics/utils/ops.py::clip_boxes
    
    Args:
        boxes (np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as (height, width).
        
    Returns:
        np.ndarray: Clipped bounding boxes.
    """
    boxes = boxes.copy()
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int], 
                ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None, 
                padding: bool = True) -> np.ndarray:
    """
    Rescale bounding boxes from one image shape to another.
    
    Source: ultralytics/utils/ops.py::scale_boxes
    原函数路径: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py
    
    Args:
        img1_shape (tuple): Shape of the source image (height, width).
        boxes (np.ndarray): Bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of ((ratio_w, ratio_h), (pad_w, pad_h)) for scaling.
        padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
        
    Returns:
        np.ndarray: Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]  # use provided ratio
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        boxes[..., 2] -= pad[0]  # x padding
        boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


class BaseOnnx(ABC):
    """ONNX模型推理基类"""
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), conf_thres: float = 0.5):
        """
        初始化ONNX模型推理器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸 (height, width)
            conf_thres (float): 置信度阈值
        """
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        
        # 使用统一的get_model_info函数获取所有模型信息，只创建一次session
        model_info = get_model_info(self.onnx_path, input_shape)
        if model_info is None:
            raise RuntimeError(f"无法加载模型: {self.onnx_path}")
        
        # 从model_info中获取所有信息
        self.session = model_info['session']
        self.input_name = model_info['input_name']
        self.output_names = model_info['output_names']
        self.input_shape = model_info['input_shape']
        self.class_names = model_info['class_names']
        
        # 记录输出形状信息
        logging.info(f"模型输出形状: {model_info['output_shape']}")
    
    
    @abstractmethod
    def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs) -> List[np.ndarray]:
        """后处理抽象方法，子类需要实现"""
        pass
    
    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None, **kwargs) -> Tuple[List[np.ndarray], tuple]:
        """
        对图像进行推理
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            Tuple[List[np.ndarray], tuple]: 检测结果列表和原始图像形状
        """
        # 预处理
        preprocess_result = self._preprocess(image)
        if len(preprocess_result) == 3:
            # 兼容旧版本返回值 (input_tensor, scale, original_shape)
            input_tensor, scale, original_shape = preprocess_result
            ratio_pad = None
        else:
            # 新版本返回值 (input_tensor, scale, original_shape, ratio_pad)
            input_tensor, scale, original_shape, ratio_pad = preprocess_result
        
        # 检查模型期望的batch维度并调整输入
        model_input_shape = self.session.get_inputs()[0].shape
        expected_batch_size = model_input_shape[0] if isinstance(model_input_shape[0], int) and model_input_shape[0] > 0 else 1
        
        if expected_batch_size > 1 and input_tensor.shape[0] == 1:
            # 如果模型期望batch>1，但输入是batch=1，则复制输入以满足要求
            input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)
            logging.debug(f"调整输入batch维度从1到{expected_batch_size}")
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理 - 根据子类不同传递不同参数
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        
        # RF-DETR需要完整的outputs，其他模型使用第一个输出
        if type(self).__name__ == 'RFDETROnnx':
            detections = self._postprocess(outputs, effective_conf_thres, **kwargs)
        else:
            prediction = outputs[0]
            detections = self._postprocess(prediction, effective_conf_thres, scale=scale, ratio_pad=ratio_pad, **kwargs)
        
        # 如果输入是多batch但只处理一张图片，只返回第一个batch的结果
        if (expected_batch_size > 1 and len(detections) > 1):
            detections = [detections[0]]
            logging.debug(f"只返回第一个batch的检测结果")
        
        return detections, original_shape
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """预处理图像"""
        return preprocess_image(image, self.input_shape)


class YoloOnnx(BaseOnnx):
    """
    传统YOLO模型ONNX推理类 (原DetONNX)
    
    支持YOLOv5、YOLOv8等使用NMS后处理的YOLO模型
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), 
                 conf_thres: float = 0.5, iou_thres: float = 0.5, 
                 multi_label: bool = True, use_ultralytics_preprocess: bool = True,
                 has_objectness: bool = False):
        """
        初始化YOLO检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
            multi_label (bool): 是否允许多标签检测，默认True
            use_ultralytics_preprocess (bool): 是否使用Ultralytics兼容的预处理
            has_objectness (bool): 模型是否有objectness分支，默认False（适应现代YOLO）
        """
        super().__init__(onnx_path, input_shape, conf_thres)
        self.iou_thres = iou_thres
        self.multi_label = multi_label
        self.use_ultralytics_preprocess = use_ultralytics_preprocess
        self.has_objectness = has_objectness
        self.model_type = self._detect_model_type()  # 自动检测模型类型
    
    def _detect_model_type(self) -> str:
        """
        自动检测YOLO模型类型
        
        Returns:
            str: 模型类型 "yolo"
        """
        # 验证输出形状来确定模型类型
        dummy_input = np.random.randn(1, 3, self.input_shape[0], self.input_shape[1]).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: dummy_input})
        output_shape = outputs[0].shape
        
        # YOLO模型通常输出 [batch, num_anchors, 4 + 1 + num_classes] 或 [batch, num_anchors, 4 + num_classes]
        if len(output_shape) == 3:
            num_features = output_shape[2]
            if num_features > 5:  # 至少有bbox(4) + objectness(1) 或 bbox(4) + classes(>=1)
                logging.info(f"检测到YOLO模型，输出形状: {output_shape}")
                return "yolo"
        
        return "yolo"  # 默认返回yolo
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple, Optional[tuple]]:
        """
        YOLO预处理，支持Ultralytics兼容模式
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            
        Returns:
            Tuple: (input_tensor, scale, original_shape, ratio_pad)
        """
        if self.use_ultralytics_preprocess:
            # 使用Ultralytics兼容的预处理，返回ratio_pad信息
            from utils.image_processing import UltralyticsLetterBox
            letterbox = UltralyticsLetterBox(new_shape=self.input_shape)
            input_tensor, scale, original_shape, ratio_pad = letterbox(image)
            return input_tensor, scale, original_shape, (((scale, scale), ratio_pad))
        else:
            # 使用原始预处理方法
            input_tensor, scale, original_shape = preprocess_image(image, self.input_shape)
            return input_tensor, scale, original_shape, None
    
    def _postprocess(self, prediction: np.ndarray, conf_thres: float, scale: float = 1.0, 
                    ratio_pad: Optional[tuple] = None, iou_thres: Optional[float] = None) -> List[np.ndarray]:
        """
        YOLO模型后处理，包含NMS
        
        Args:
            prediction (np.ndarray): 模型原始输出
            conf_thres (float): 置信度阈值
            scale (float): 图像缩放因子
            ratio_pad (Optional[tuple]): Ratio和padding信息 (((ratio_w, ratio_h), (pad_w, pad_h)))
            iou_thres (Optional[float]): IoU阈值
            
        Returns:
            List[np.ndarray]: 后处理后的检测结果
        """
        # 自适应处理YOLO输出格式
        # YOLO官方库输出: [B, bbox+C, N] 例如 (1, 84, 8400)
        # 我们的处理逻辑: [B, N, bbox+C] 例如 (1, 8400, 84)
        original_shape = prediction.shape
        
        # 鲁棒的维度判断逻辑：
        # 1. 如果第二维小于第三维，且第二维在合理的特征数范围内(4-200)，则可能是[B, C, N]格式
        # 2. 特征数应该是4+类别数，一般在80-200之间比较合理
        if (prediction.shape[1] < prediction.shape[2] and 
            4 <= prediction.shape[1] <= 200 and
            prediction.shape[2] > prediction.shape[1]):
            # 转换为我们期望的格式: [B, N, C]
            prediction = prediction.transpose(0, 2, 1)
            logging.info(f"YOLO输出格式自适应转换: {original_shape} -> {prediction.shape}")
        
        # 验证最终格式的合理性
        if prediction.shape[2] < 4:
            raise ValueError(f"YOLO输出格式异常: {prediction.shape}，特征维度应该至少包含4个bbox坐标")
        
        # 检查坐标格式并转换为像素坐标（对齐Ultralytics实现）
        bbox_coords = prediction[..., :4]
        max_coord = np.max(np.abs(bbox_coords))
        
        if max_coord <= 1.0:
            # 归一化坐标，需要转换为像素坐标
            prediction[..., 0] *= self.input_shape[1]  # x_center
            prediction[..., 1] *= self.input_shape[0]  # y_center
            prediction[..., 2] *= self.input_shape[1]  # width
            prediction[..., 3] *= self.input_shape[0]  # height
        else:
            # 已经是像素坐标，无需转换
            pass
        
        # NMS后处理，传递multi_label、model_type和objectness信息
        effective_iou_thres = iou_thres if iou_thres is not None else self.iou_thres
        detections = non_max_suppression(
            prediction, 
            conf_thres=conf_thres, 
            iou_thres=effective_iou_thres,
            multi_label=self.multi_label,
            model_type=self.model_type,
            has_objectness=self.has_objectness
        )
        
        # 使用Ultralytics风格的坐标还原
        if detections and len(detections[0]) > 0:
            if ratio_pad is not None and self.use_ultralytics_preprocess:
                # 使用scale_boxes进行精确的坐标还原
                # 从输入尺寸坐标还原到原图坐标
                boxes = detections[0][:, :4].copy()
                original_shape = (int(self.input_shape[0] / ratio_pad[0][0]), 
                                int(self.input_shape[1] / ratio_pad[0][1]))
                
                # 使用scale_boxes函数进行坐标还原
                scaled_boxes = scale_boxes(
                    img1_shape=self.input_shape,  # 输入尺寸
                    boxes=boxes,                   # 检测框坐标
                    img0_shape=original_shape,     # 原图尺寸
                    ratio_pad=ratio_pad,           # ratio和padding信息
                    padding=True                   # 使用padding
                )
                
                # 更新检测结果中的坐标
                detections[0][:, :4] = scaled_boxes
            else:
                # 回退到简单的缩放方法
                detections[0][:, :4] /= scale
            
        return detections
    
    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None, 
                 iou_thres: Optional[float] = None) -> Tuple[List[np.ndarray], tuple]:
        """
        YOLO推理接口
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值
            iou_thres (Optional[float]): IoU阈值
            
        Returns:
            Tuple[List[np.ndarray], tuple]: 检测结果列表和原始图像形状
        """
        return super().__call__(image, conf_thres=conf_thres, iou_thres=iou_thres)


class RTDETROnnx(YoloOnnx):
    """
    RT-DETR模型ONNX推理类 (原YoloRTDETROnnx)
    
    完全复刻ultralytics RTDETRValidator的后处理逻辑，端到端检测，无需NMS
    模型输出格式: [batch, 300, 19] = [batch, queries, (4_bbox + 15_classes)]
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), 
                 conf_thres: float = 0.001, iou_thres: float = 0.5):
        """
        初始化RT-DETR检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值，默认0.001
            iou_thres (float): IoU阈值，RT-DETR不使用，保持接口统一
        """
        # 调用BaseOnnx初始化，跳过YoloOnnx的iou_thres设置
        BaseOnnx.__init__(self, onnx_path, input_shape, conf_thres)
        
        # RT-DETR输出格式验证已在BaseOnnx的get_model_info中完成
        # 这里可以添加额外的RT-DETR特定验证逻辑
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """
        RT-DETR预处理（复刻ultralytics风格，直接resize不保持长宽比）
        
        Source: ultralytics/data/base.py::BaseDataset.load_image
        参考: ultralytics/models/rtdetr/val.py::RTDETRDataset.build_transforms
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            
        Returns:
            Tuple: (预处理后的tensor, scale_h, scale_w, 原始形状)
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        target_h, target_w = self.input_shape
        
        # 直接resize到目标尺寸，不保持长宽比（与ultralytics一致）
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为RGB（ultralytics通常使用RGB）
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        normalized = resized_rgb.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        # 计算缩放因子（用于最终的坐标还原）
        scale_h = target_h / h
        scale_w = target_w / w
        
        # RT-DETR直接拉伸，需要返回实际的缩放因子
        # 返回(scale_w, scale_h)作为scale，在后处理中使用
        scale = (scale_w, scale_h)
        
        return tensor, scale, original_shape
    
    def _postprocess(self, preds: np.ndarray, conf_thres: float, **_kwargs) -> List[np.ndarray]:
        """
        RT-DETR后处理（完全复刻ultralytics RTDETRValidator.postprocess）
        
        Source: ultralytics/models/rtdetr/val.py::RTDETRValidator.postprocess
        原函数路径: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L157-L188
        
        Args:
            preds (np.ndarray): 模型原始输出 [batch, 300, 19]
            conf_thres (float): 置信度阈值
            **kwargs: 其他参数（RT-DETR不使用scale等）
            
        Returns:
            List[np.ndarray]: 检测结果列表，坐标在输入图像尺寸上
        """
        # 处理预测格式（复刻 ultralytics/models/rtdetr/val.py#L173-L174）
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        
        bs, _, _ = preds[0].shape  # batch_size, num_queries(300), num_features(19)
        
        # 分割bbox和scores（复刻 ultralytics/models/rtdetr/val.py#L177）
        bboxes = preds[0][:, :, :4]    # [batch, 300, 4] - bbox坐标
        scores = preds[0][:, :, 4:]    # [batch, 300, 15] - 类别分数
        
        # 缩放bbox到输入图像尺寸（复刻 ultralytics/models/rtdetr/val.py#L178）
        # RT-DETR输出的bbox是归一化坐标[0,1]，需要乘以输入尺寸转换为像素坐标
        imgsz = self.input_shape[0]  # ultralytics假设输入是正方形
        bboxes = bboxes * imgsz
        
        # 初始化输出
        outputs = []
        
        # 为每个batch中的图像处理
        for i in range(bs):
            bbox = bboxes[i]  # [300, 4]
            score_matrix = scores[i]  # [300, 15]
            
            # 坐标转换从xywh到xyxy（复刻 ultralytics/models/rtdetr/val.py#L181）
            bbox = xywh2xyxy(bbox)
            
            # 获取每个query的最大类别分数和索引（复刻 ultralytics/models/rtdetr/val.py#L182）
            score = np.max(score_matrix, axis=-1)  # [300,] - 最大分数
            cls = np.argmax(score_matrix, axis=-1)  # [300,] - 类别索引
            
            # 组合预测结果（复刻 ultralytics/models/rtdetr/val.py#L183）
            pred = np.column_stack([bbox, score, cls])  # [300, 6]
            
            # 按置信度排序（复刻 ultralytics/models/rtdetr/val.py#L185）
            sorted_indices = np.argsort(score)[::-1]  # 降序排序
            pred = pred[sorted_indices]
            score_sorted = score[sorted_indices]
            
            # 置信度过滤（复刻 ultralytics/models/rtdetr/val.py#L186）
            mask = score_sorted > conf_thres
            pred = pred[mask]
            
            outputs.append(pred)
        
        # 重要：RT-DETR返回的坐标是在输入图像尺寸上的，与ultralytics一致
        # 坐标缩放在后续的evaluate阶段进行
        return outputs
    


class RFDETROnnx(BaseOnnx):
    """
    RF-DETR模型ONNX推理类
    
    支持RF-DETR (ResNet-based Feature Pyramid + DETR) 模型
    输出格式：两个独立的输出 - pred_boxes 和 pred_logits
    """
    
    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (576, 576), 
                 conf_thres: float = 0.001, iou_thres: float = 0.5):
        """
        初始化RF-DETR检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸，默认576x576
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值，RF-DETR不使用，保持接口统一
        """
        super().__init__(onnx_path, input_shape, conf_thres)
        
        # 验证RF-DETR输出格式
        # 检查模型期望的batch维度
        model_input_shape = self.session.get_inputs()[0].shape
        expected_batch_size = model_input_shape[0] if isinstance(model_input_shape[0], int) and model_input_shape[0] > 0 else 1
        
        dummy_input = np.random.randn(expected_batch_size, 3, self.input_shape[0], self.input_shape[1]).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: dummy_input})
        
        if len(outputs) != 2:
            raise ValueError(f"RF-DETR模型应该有2个输出（pred_boxes, pred_logits），但实际有{len(outputs)}个")
        
        pred_boxes_shape, pred_logits_shape = outputs[0].shape, outputs[1].shape
        logging.info(f"RF-DETR输出格式 - pred_boxes: {pred_boxes_shape}, pred_logits: {pred_logits_shape}")
        
        if len(pred_boxes_shape) != 3 or pred_boxes_shape[2] != 4:
            logging.warning(f"警告: pred_boxes形状 {pred_boxes_shape} 可能不符合标准RF-DETR格式")
        
        if len(pred_logits_shape) != 3 or pred_logits_shape[1] != pred_boxes_shape[1]:
            logging.warning(f"警告: pred_logits形状 {pred_logits_shape} 与pred_boxes不匹配")
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """
        RF-DETR预处理（对齐原始实现）
        
        基于third_party/rfdetr/datasets/coco.py中的make_coco_transforms实现
        使用ImageNet标准化参数：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            
        Returns:
            Tuple: (预处理后的tensor, scale, 原始形状)
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        target_h, target_w = self.input_shape
        
        # 直接resize到目标尺寸（对齐SquareResize）
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 转换为RGB（PyTorch通常使用RGB）
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转换为[0,1]范围（对齐ToTensor）
        normalized = resized_rgb.astype(np.float32) / 255.0
        
        # 应用ImageNet标准化（对齐Normalize）
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        normalized = (normalized - imagenet_mean) / imagenet_std
        
        # 转换为CHW格式并添加batch维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        # 计算缩放因子
        scale_h = target_h / h
        scale_w = target_w / w
        # RF-DETR直接拉伸，需要返回实际的缩放因子
        scale = (scale_w, scale_h)
        
        return tensor, scale, original_shape
    
    def _postprocess(self, outputs: List[np.ndarray], conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        RF-DETR后处理逻辑（对齐原始实现）
        
        基于third_party/rfdetr/models/lwdetr.py中的PostProcess类实现
        
        Args:
            outputs (List[np.ndarray]): 模型原始输出 [pred_boxes, pred_logits]
            conf_thres (float): 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            List[np.ndarray]: 后处理后的检测结果
        """
        # 根据实际输出形状确定pred_logits和pred_boxes
        # 输出0: (4, 300, 4) - 应该是pred_boxes
        # 输出1: (4, 300, 15) - 应该是pred_logits
        if outputs[0].shape[2] == 4 and outputs[1].shape[2] > 4:
            pred_boxes, pred_logits = outputs[0], outputs[1]
        else:
            pred_logits, pred_boxes = outputs[0], outputs[1]
        
        bs = pred_boxes.shape[0]
        num_queries = pred_boxes.shape[1]
        num_classes = pred_logits.shape[2]
        
        results = []
        
        # 为每个batch中的图像处理
        for i in range(bs):
            out_logits = pred_logits[i]  # [300, 15]
            out_bbox = pred_boxes[i]     # [300, 4]
            
            # 使用sigmoid激活（对齐原始实现）
            prob = 1.0 / (1.0 + np.exp(-out_logits))  # sigmoid
            
            # topk选择（对齐原始PostProcess.forward）
            # 将prob展平并选择top-k个值
            prob_flat = prob.flatten()  # [300*15]
            
            # 选择top 300个检测（num_select=300）
            num_select = min(300, len(prob_flat))
            topk_indices = np.argpartition(prob_flat, -num_select)[-num_select:]
            topk_indices = topk_indices[np.argsort(prob_flat[topk_indices])][::-1]  # 降序排序
            
            scores = prob_flat[topk_indices]
            
            # 计算对应的box和label索引
            topk_boxes_idx = topk_indices // num_classes  # 对应的query索引
            labels = topk_indices % num_classes           # 对应的类别索引
            
            # 提取对应的bbox
            boxes = out_bbox[topk_boxes_idx]  # [num_select, 4]
            
            # 坐标转换：从cxcywh到xyxy（对齐box_ops.box_cxcywh_to_xyxy）
            x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            w = np.clip(w, a_min=0.0, a_max=None)  # clamp(min=0.0)
            h = np.clip(h, a_min=0.0, a_max=None)
            
            x1 = x_c - 0.5 * w
            y1 = y_c - 0.5 * h  
            x2 = x_c + 0.5 * w
            y2 = y_c + 0.5 * h
            
            boxes_xyxy = np.column_stack([x1, y1, x2, y2])
            
            # 缩放到输入图像尺寸（从归一化坐标[0,1]到像素坐标）
            imgsz_w, imgsz_h = self.input_shape[1], self.input_shape[0]  # (width, height)
            scale_fct = np.array([imgsz_w, imgsz_h, imgsz_w, imgsz_h])
            boxes_xyxy = boxes_xyxy * scale_fct
            
            # 置信度过滤
            mask = scores > conf_thres
            if np.any(mask):
                final_scores = scores[mask]
                final_labels = labels[mask]
                final_boxes = boxes_xyxy[mask]
                
                # 组合结果：[x1, y1, x2, y2, conf, class]
                pred = np.column_stack([final_boxes, final_scores, final_labels])
            else:
                pred = np.zeros((0, 6))
            
            results.append(pred)
        
        return results
    


def create_detector(model_type: str, onnx_path: str, **kwargs) -> BaseOnnx:
    """
    工厂函数：根据模型类型创建相应的检测器
    
    Args:
        model_type (str): 模型类型，支持 'yolo', 'rtdetr', 'rfdetr'
        onnx_path (str): ONNX模型路径
        **kwargs: 其他参数
        
    Returns:
        BaseOnnx: 相应的检测器实例
        
    Raises:
        ValueError: 不支持的模型类型
    """
    model_type = model_type.lower()
    
    if model_type in ['yolo', 'yolov5', 'yolov8']:
        return YoloOnnx(onnx_path, **kwargs)
    elif model_type in ['rtdetr', 'rt-detr']:
        return RTDETROnnx(onnx_path, **kwargs)
    elif model_type in ['rfdetr', 'rf-detr']:
        return RFDETROnnx(onnx_path, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 添加通用的数据集评估功能
class DatasetEvaluator:
    """通用数据集评估器"""
    
    def __init__(self, detector: BaseOnnx):
        """
        初始化评估器
        
        Args:
            detector (BaseOnnx): 检测器实例
        """
        self.detector = detector
    
    def evaluate_dataset(
        self, 
        dataset_path: str,
        output_transform: Optional[Callable] = None,
        conf_threshold: float = 0.25,  # 与Ultralytics验证模式对齐，避免过低阈值产生大量误检
        iou_threshold: float = 0.7,  # 保留参数以保持一致性
        max_images: Optional[int] = None,
        exclude_files: Optional[List[str]] = None,  # 允许用户指定需要排除的文件
        exclude_labels_containing: Optional[List[str]] = None  # 允许用户指定需要排除的标签内容
    ) -> Dict[str, Any]:
        """
        在YOLO格式数据集上评估模型性能
        
        Args:
            dataset_path (str): 数据集路径
            output_transform (Optional[Callable]): 输出转换函数
            conf_threshold (float): 置信度阈值，默认0.25与Ultralytics对齐
                注意：Ultralytics在验证模式下会将默认的0.001重置为0.25
                参考：ultralytics/utils/metrics.py:403 v8.3.179
                conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
            iou_threshold (float): IoU阈值
            max_images (Optional[int]): 最大评估图像数量
            exclude_files (Optional[List[str]]): 需要排除的文件名列表
            exclude_labels_containing (Optional[List[str]]): 排除包含指定内容的标签文件
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 重要说明：置信度阈值默认值更改
        # ==========================================
        # 之前版本使用 conf_threshold=0.001，但发现与Ultralytics结果存在显著差异
        # 原因：Ultralytics在验证过程中会自动将过低的置信度阈值重置为0.25
        # 
        # 具体机制（参考ultralytics/utils/metrics.py:403）：
        # conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
        # 
        # 即：如果传入的conf是默认验证值0.001，会被强制重置为预测模式的0.25
        # 
        # 为了保持与Ultralytics一致的评估结果，现将默认值统一设置为0.25
        # 这样可以避免因置信度阈值差异导致的指标差异（P/R/mAP等）
        # ==========================================
        
        dataset_path = Path(dataset_path)
        
        # 数据集路径检测逻辑
        test_images_dir = dataset_path / "images" / "test"
        test_labels_dir = dataset_path / "labels" / "test"
        val_images_dir = dataset_path / "images" / "val"
        val_labels_dir = dataset_path / "labels" / "val"
        
        if test_images_dir.exists() and test_labels_dir.exists():
            images_dir = test_images_dir
            labels_dir = test_labels_dir
            split_name = "test"
            logging.info("使用test数据集进行评估")
        elif val_images_dir.exists() and val_labels_dir.exists():
            images_dir = val_images_dir
            labels_dir = val_labels_dir
            split_name = "val"
            logging.info("使用val数据集进行评估")
        else:
            # 回退逻辑
            images_dir = dataset_path / "images" / "train"
            labels_dir = dataset_path / "labels" / "train"
            if images_dir.exists() and labels_dir.exists():
                split_name = "train"
                logging.info("使用train数据集进行评估")
            else:
                images_dir = dataset_path / "images"
                labels_dir = dataset_path / "labels" 
                split_name = "root"
                if not images_dir.exists():
                    raise ValueError(f"未找到有效的图像目录")
                if not labels_dir.exists():
                    raise ValueError(f"标签目录不存在: {labels_dir}")
                logging.info("使用根目录下的images/labels进行评估")
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        # 检查图像文件是否可访问，过滤损坏或无效的文件
        valid_image_files = []
        exclude_files = exclude_files or []
        exclude_labels_containing = exclude_labels_containing or []
        
        for image_file in image_files:
            # 检查是否在排除列表中
            if image_file.name in exclude_files:
                continue
                
            # 基本有效性检查：文件存在且大小大于0
            if not (image_file.exists() and image_file.stat().st_size > 0):
                continue
                
            # 检查是否有对应的标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                continue
                
            # 检查标签文件内容是否包含需要排除的内容
            if exclude_labels_containing:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if any(exclude_text in content for exclude_text in exclude_labels_containing):
                            continue
                except Exception:
                    continue
                    
            valid_image_files.append(image_file)
        
        image_files = valid_image_files
        
        if not image_files:
            logging.warning(f"在 {images_dir} 中未找到有效的图像文件")
        
        if max_images:
            image_files = image_files[:max_images]
        
        logging.info(f"开始评估{split_name}数据集，共 {len(image_files)} 张图像")
        
        predictions = []
        ground_truths = []
        
        # 加载类别名称
        names = {}
        data_yaml = dataset_path / "classes.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                names = data_config.get('names', {})
                if isinstance(names, list):
                    names = {i: name for i, name in enumerate(names)}
        
        # 性能统计
        times = {
            'preprocess': [],
            'inference': [],
            'postprocess': []
        }
        
        logging.info(f"评估 {len(image_files)} 张图像...")

        for i, image_file in enumerate(image_files):
            if i % 100 == 0:
                logging.info(f"处理进度: {i}/{len(image_files)}")
            
            # 读取图像
            start_time = time.time()
            image = cv2.imread(str(image_file))
            if image is None:
                logging.warning(f"无法读取图像: {image_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # 测量预处理时间
            preprocess_start = time.time()
            # 进行检测
            detections, original_shape = self.detector(image, conf_thres=conf_threshold)
            inference_end = time.time()
            
            # 应用输出转换（如果提供）
            if output_transform is not None:
                detections = output_transform(detections, original_shape)
            
            postprocess_end = time.time()
            
            # 记录时间
            times['preprocess'].append((preprocess_start - start_time) * 1000)
            times['inference'].append((inference_end - preprocess_start) * 1000)
            times['postprocess'].append((postprocess_end - inference_end) * 1000)
            
            # 处理检测结果（坐标已在模型后处理中正确缩放）
            if detections and len(detections[0]) > 0:
                pred = detections[0].copy()  # [N, 6] format: [x1, y1, x2, y2, conf, class]
                
                # 对于使用了新坐标处理逻辑的YOLO模型，检测结果已经在原图坐标系
                # 对于RT-DETR和RF-DETR等特殊模型，仍需要额外缩放
                if type(self.detector).__name__ in ['RTDETROnnx', 'RFDETROnnx']:
                    pred[:, [0, 2]] = pred[:, [0, 2]] * img_width / self.detector.input_shape[1]   # x坐标缩放
                    pred[:, [1, 3]] = pred[:, [1, 3]] * img_height / self.detector.input_shape[0]  # y坐标缩放
                # YoloOnnx使用新的坐标处理逻辑，坐标已正确缩放，无需额外处理
            else:
                pred = np.zeros((0, 6))
            
            predictions.append(pred)
            
            # 安全加载标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            gt = self._load_yolo_labels_safe(str(label_file), img_width, img_height)
            ground_truths.append(gt)
        
        # 计算指标
        results = evaluate_detection(predictions, ground_truths, names)
        
        # 添加性能统计
        if times['preprocess']:
            results['speed_preprocess'] = np.mean(times['preprocess'])
            results['speed_inference'] = np.mean(times['inference']) 
            results['speed_loss'] = 0.0
            results['speed_postprocess'] = np.mean(times['postprocess'])
        
        # 打印结果
        print_metrics(results, names)
        
        return results
    
    def _load_yolo_labels_safe(self, label_path: str, img_width: int, img_height: int) -> np.ndarray:
        """安全加载YOLO标签，跳过无效行"""
        if not Path(label_path).exists():
            return np.zeros((0, 5))
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # 转换为xyxy格式
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        labels.append([class_id, x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        # 跳过无效行
                        continue
        
        return np.array(labels) if labels else np.zeros((0, 5))


# 为了向后兼容，保留旧的类名
DetONNX = YoloOnnx
YoloRTDETROnnx = RTDETROnnx