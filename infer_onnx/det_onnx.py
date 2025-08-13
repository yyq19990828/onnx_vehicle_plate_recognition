import onnxruntime
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from pathlib import Path
import cv2
import yaml

from .utils import preload_onnx_libraries, get_best_available_providers
from utils.image_processing import preprocess_image
from utils.nms import non_max_suppression
from utils.detection_metrics import evaluate_detection, load_yolo_labels, print_metrics

class DetONNX:
    """
    A class for performing object detection using an ONNX model.
    
    This class supports Ultralytics YOLO models exported to ONNX format.
    Compatible with YOLOv8, YOLOv5, and other Ultralytics-based models.
    """

    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), conf_thres: float = 0.5, iou_thres: float = 0.5):
        # Ensure ONNX Runtime libraries are preloaded if necessary
        preload_onnx_libraries()

        """
        Initializes the DetONNX instance.

        Args:
            onnx_path (str): The path to the ONNX model file.
            input_shape (Tuple[int, int]): The expected input shape (height, width) of the model.
            conf_thres (float): Confidence threshold for NMS.
            iou_thres (float): IoU threshold for NMS.
        """
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Create an ONNX Runtime session using the best available providers
        providers = get_best_available_providers(self.onnx_path)
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 优先从ONNX模型中读取输入形状，如果是动态的则使用默认值
        model_input_shape = self.session.get_inputs()[0].shape
        if (len(model_input_shape) >= 4 and 
            isinstance(model_input_shape[2], int) and model_input_shape[2] > 0 and
            isinstance(model_input_shape[3], int) and model_input_shape[3] > 0):
            # 如果模型输入形状是固定的 (batch, channels, height, width)
            self.input_shape = (model_input_shape[2], model_input_shape[3])
            logging.info(f"从ONNX模型读取到固定输入形状: {self.input_shape}")
        else:
            # 如果是动态形状（包含字符串或负数），使用传入的默认值
            self.input_shape = input_shape
            logging.info(f"模型输入形状为动态 {model_input_shape}，使用默认形状: {self.input_shape}")

    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None) -> Tuple[List[np.ndarray], tuple]:
        """
        Performs inference on a single image.

        Args:
            image (np.ndarray): The input image in BGR format.
            conf_thres (Optional[float]): Confidence threshold, if None uses self.conf_thres

        Returns:
            Tuple[List[np.ndarray], tuple]: A tuple containing the list of detections and the original image shape.
                                            Each detection is an array of shape [num_detections, 6]
                                            representing (x1, y1, x2, y2, conf, class_id).
        """
        # Preprocess the image
        input_tensor, scale, original_shape = preprocess_image(image, self.input_shape)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # The primary output is usually the first one
        prediction = outputs[0]

        # The model's output format is [center_x, center_y, width, height, ...],
        # where coordinates are normalized.
        # We need to scale them to the input image size (e.g., 640x640).
        prediction[..., 0] *= self.input_shape[1]  # x_center
        prediction[..., 1] *= self.input_shape[0]  # y_center
        prediction[..., 2] *= self.input_shape[1]  # width
        prediction[..., 3] *= self.input_shape[0]  # height

        # 使用传入的置信度阈值，如果没有则使用默认值
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        effective_iou_thres = self.iou_thres  # IoU阈值保持不变
        
        # Post-process the prediction with NMS
        detections = non_max_suppression(
            prediction,
            conf_thres=effective_conf_thres,
            iou_thres=effective_iou_thres
        )

        # Scale boxes back to original image size
        if detections and len(detections[0]) > 0:
            detections[0][:, :4] /= scale

        return detections, original_shape

    def evaluate_dataset(
        self, 
        dataset_path: str,
        output_transform: Optional[Callable] = None,
        conf_threshold: float = 0.001,
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        在YOLO格式数据集上评估模型性能
        
        Args:
            dataset_path (str): 数据集路径，应包含images和labels文件夹
            output_transform (Optional[Callable]): 输出转换函数，将模型输出转换为YOLO格式
                                                 函数签名: (detections, original_shape) -> detections
            conf_threshold (float): 置信度阈值，低于此阈值的检测将被过滤
            max_images (Optional[int]): 最大评估图像数量，None表示评估所有图像
            
        Returns:
            Dict[str, Any]: 评估结果，包含mAP、precision、recall等指标
        """
        dataset_path = Path(dataset_path)
        
        # 优先检测test文件夹，如果没有则使用val文件夹
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
            # 回退到原始的images/labels结构（train数据集）
            images_dir = dataset_path / "images" / "train"
            labels_dir = dataset_path / "labels" / "train"
            if images_dir.exists() and labels_dir.exists():
                split_name = "train"
                logging.info("使用train数据集进行评估")
            else:
                # 最终回退到根目录的images/labels
                images_dir = dataset_path / "images"
                labels_dir = dataset_path / "labels" 
                split_name = "root"
                if not images_dir.exists():
                    raise ValueError(f"未找到有效的图像目录。请确保数据集包含以下结构之一:\n"
                                   f"1. {dataset_path}/images/test/ 和 {dataset_path}/labels/test/\n"
                                   f"2. {dataset_path}/images/val/ 和 {dataset_path}/labels/val/\n"
                                   f"3. {dataset_path}/images/train/ 和 {dataset_path}/labels/train/\n"
                                   f"4. {dataset_path}/images/ 和 {dataset_path}/labels/")
                if not labels_dir.exists():
                    raise ValueError(f"标签目录不存在: {labels_dir}")
                logging.info("使用根目录下的images/labels进行评估")
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logging.info(f"开始评估{split_name}数据集，共 {len(image_files)} 张图像")
        
        predictions = []
        ground_truths = []
        
        # 加载类别名称（如果存在classes.yaml）
        names = {}
        data_yaml = dataset_path / "classes.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                names = data_config.get('names', {})
                if isinstance(names, list):
                    names = {i: name for i, name in enumerate(names)}
        
        for i, image_file in enumerate(image_files):
            if i % 100 == 0:
                logging.info(f"处理进度: {i}/{len(image_files)}")
            
            # 读取图像
            image = cv2.imread(str(image_file))
            if image is None:
                logging.warning(f"无法读取图像: {image_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # 进行检测，传递置信度阈值
            detections, original_shape = self(image, conf_thres=conf_threshold)
            
            # 应用输出转换（如果提供）
            if output_transform is not None:
                detections = output_transform(detections, original_shape)
            
            # 处理检测结果（已经在__call__中进行了置信度过滤）
            if detections and len(detections[0]) > 0:
                pred = detections[0]
            else:
                pred = np.zeros((0, 6))
            
            predictions.append(pred)
            
            # 加载对应的标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            gt = load_yolo_labels(str(label_file), img_width, img_height)
            ground_truths.append(gt)
        
        # 计算指标
        results = evaluate_detection(predictions, ground_truths, names)
        
        # 打印结果
        print_metrics(results, names)
        
        return results


class RFDETROnnx(DetONNX):
    """
    Roboflow RF-DETR模型的专用检测类，继承自DetONNX。
    
    此类专门支持Roboflow的RF-DETR模型，RF-DETR使用不同的输出格式：
    - pred_boxes: 归一化的center_x, center_y, width, height坐标
    - pred_logits: 分类得分（使用sigmoid激活）
    
    与Ultralytics YOLO模型的主要区别：
    1. 输入尺寸默认为576x576而不是640x640
    2. 输出格式为两个独立的张量而不是单个张量
    3. 坐标格式为cx,cy,w,h而不是xyxy
    4. 不需要额外的NMS后处理
    """

    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (576, 576), conf_thres: float = 0.5, iou_thres: float = 0.5):
        """
        初始化RF-DETR检测器。
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸，RF-DETR默认为(576, 576)
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值（RF-DETR通常不需要额外的NMS）
        """
        super().__init__(onnx_path, input_shape, conf_thres, iou_thres)
        
        # RF-DETR的输出名称
        self.pred_boxes_name = "pred_boxes"
        self.pred_logits_name = "pred_logits"
        
        # 验证输出名称
        actual_output_names = [output.name for output in self.session.get_outputs()]
        if self.pred_boxes_name not in actual_output_names or self.pred_logits_name not in actual_output_names:
            logging.warning(f"期望的输出名称 {[self.pred_boxes_name, self.pred_logits_name]} 与实际输出 {actual_output_names} 不匹配")

    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None) -> Tuple[List[np.ndarray], tuple]:
        """
        使用RF-DETR模型进行推理。
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值，如果为None则使用self.conf_thres
            
        Returns:
            Tuple[List[np.ndarray], tuple]: 检测结果列表和原始图像形状
                                          每个检测结果格式为 [x1, y1, x2, y2, conf, class_id]
        """
        # 使用RF-DETR专用的预处理
        input_tensor, scale, original_shape, offset = self._preprocess_rfdetr(image)
        
        # 运行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # RF-DETR的输出格式
        pred_boxes = outputs[0]  # shape: (1, num_detections, 4) - 归一化的cx,cy,w,h坐标
        pred_logits = outputs[1]  # shape: (1, num_detections, num_classes) - 分类得分
        
        # 处理批次维度
        pred_boxes = pred_boxes[0]  # (num_detections, 4)
        pred_logits = pred_logits[0]  # (num_detections, num_classes)
        
        # 使用传入的置信度阈值，如果没有则使用默认值
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        
        # 使用RF-DETR专用的后处理
        detections_array = self._postprocess_rfdetr(
            pred_boxes, pred_logits, scale, original_shape, offset, effective_conf_thres
        )
        
        detections = [detections_array] if len(detections_array) > 0 else [[]]
        
        return detections, original_shape
    
    def _preprocess_rfdetr(self, image: np.ndarray) -> tuple:
        """
        RF-DETR专用的预处理函数，匹配原始RF-DETR的SquareResize + ImageNet标准化
        
        Args:
            image: 输入图像 (BGR格式)
        
        Returns:
            预处理后的tensor, 缩放因子, 原始形状, 偏移量
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        target_size = self.input_shape[0]  # 假设是正方形输入
        
        # 使用直接方形缩放（SquareResize），不保持长宽比，匹配原始RF-DETR
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # 转换为RGB（RF-DETR需要RGB输入）
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        resized_float = resized_rgb.astype(np.float32) / 255.0
        
        # 应用ImageNet标准化，匹配原始RF-DETR训练时的标准化参数
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = (resized_float - mean) / std
        
        # 转换为CHW格式并添加batch维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        # 对于直接方形缩放，计算每个维度的缩放因子
        scale_h = target_size / h
        scale_w = target_size / w
        scale = (scale_h, scale_w)  # 返回元组而不是单一值
        
        # 没有偏移量，因为是直接缩放
        offset = (0, 0)
        
        return tensor, scale, original_shape, offset
    
    def _postprocess_rfdetr(
        self, 
        pred_boxes: np.ndarray, 
        pred_logits: np.ndarray,
        scale: tuple,
        original_shape: tuple,
        offset: tuple,
        conf_threshold: float,
        num_select: int = 300
    ) -> np.ndarray:
        """
        RF-DETR专用的后处理函数，匹配原始RF-DETR的TopK选择机制
        
        Args:
            pred_boxes: 预测的边界框 (num_queries, 4) - 归一化的 [cx, cy, w, h]
            pred_logits: 预测的logits (num_queries, num_classes)
            scale: 预处理时的缩放因子 (scale_h, scale_w)
            original_shape: 原始图像形状 (H, W)
            offset: 预处理时的偏移量 (x_offset, y_offset) - 直接缩放时为(0,0)
            conf_threshold: 置信度阈值
            num_select: TopK选择的数量，匹配RF-DETR的num_select=300
        
        Returns:
            检测结果 (N, 6) - [x1, y1, x2, y2, conf, class]
        """
        scale_h, scale_w = scale
        target_size = self.input_shape[0]
        
        # 应用sigmoid激活
        scores = 1 / (1 + np.exp(-pred_logits))
        
        # 实现RF-DETR的TopK选择机制
        # 将scores展平为(num_queries * num_classes,)
        scores_flat = scores.reshape(-1)
        num_queries, num_classes = scores.shape
        
        # 选择前num_select个最高得分
        num_select = min(num_select, len(scores_flat))
        topk_indices = np.argpartition(scores_flat, -num_select)[-num_select:]
        topk_values = scores_flat[topk_indices]
        
        # 过滤低置信度检测
        valid_mask = topk_values > conf_threshold
        if not np.any(valid_mask):
            return np.zeros((0, 6))
        
        # 应用有效性掩码
        topk_indices = topk_indices[valid_mask]
        topk_values = topk_values[valid_mask]
        
        # 将平坦索引转换回(query_idx, class_idx)
        topk_boxes_idx = topk_indices // num_classes  # 查询索引
        class_ids = topk_indices % num_classes         # 类别索引
        
        # 获取对应的边界框
        selected_boxes = pred_boxes[topk_boxes_idx]
        selected_scores = topk_values
        selected_classes = class_ids
        
        # 转换归一化坐标到像素坐标（相对于target_size）
        cx = selected_boxes[:, 0] * target_size
        cy = selected_boxes[:, 1] * target_size
        w = selected_boxes[:, 2] * target_size
        h = selected_boxes[:, 3] * target_size
        
        # 转换为x1,y1,x2,y2格式（相对于target_size）
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 对于直接缩放，直接转换回原始图像坐标
        # 使用不同的缩放因子处理x和y坐标
        x1 /= scale_w
        y1 /= scale_h
        x2 /= scale_w
        y2 /= scale_h
        
        # 裁剪到原始图像边界
        h_orig, w_orig = original_shape
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)
        
        # 组合检测结果
        detections = np.column_stack([x1, y1, x2, y2, selected_scores, selected_classes])
        
        return detections