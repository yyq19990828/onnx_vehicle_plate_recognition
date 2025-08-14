"""
YOLO格式的RT-DETR ONNX推理类
完全复刻ultralytics的RTDETRValidator后处理逻辑
"""

import onnxruntime
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from pathlib import Path
import cv2
import yaml
import time

from .utils import preload_onnx_libraries, get_best_available_providers
from utils.image_processing import preprocess_image
from utils.detection_metrics import evaluate_detection, load_yolo_labels, print_metrics


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    与ultralytics完全一致的坐标转换函数
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.
    
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


class YoloRTDETROnnx:
    """
    专门用于YOLO格式RT-DETR模型的ONNX推理类
    完全复刻ultralytics RTDETRValidator的后处理逻辑
    
    模型输出格式: [batch, 300, 19] = [batch, queries, (4_bbox + 15_classes)]
    """

    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), conf_thres: float = 0.001):
        """
        初始化YOLO格式RT-DETR检测器
        
        Args:
            onnx_path (str): ONNX模型文件路径
            input_shape (Tuple[int, int]): 输入图像尺寸
            conf_thres (float): 置信度阈值，默认0.001
        """
        # 确保ONNX Runtime库被预加载
        preload_onnx_libraries()
        
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        
        # 创建ONNX Runtime会话
        providers = get_best_available_providers(self.onnx_path)
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 从ONNX模型中读取输入形状
        model_input_shape = self.session.get_inputs()[0].shape
        if (len(model_input_shape) >= 4 and 
            isinstance(model_input_shape[2], int) and model_input_shape[2] > 0 and
            isinstance(model_input_shape[3], int) and model_input_shape[3] > 0):
            self.input_shape = (model_input_shape[2], model_input_shape[3])
            logging.info(f"从ONNX模型读取到固定输入形状: {self.input_shape}")
        else:
            self.input_shape = input_shape
            logging.info(f"模型输入形状为动态 {model_input_shape}，使用默认形状: {self.input_shape}")
        
        # 验证模型输出维度
        dummy_input = np.random.randn(1, 3, self.input_shape[0], self.input_shape[1]).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: dummy_input})
        output_shape = outputs[0].shape
        logging.info(f"模型输出形状: {output_shape}")
        
        # 验证是否为YOLO格式RT-DETR (期望: [1, 300, 19])
        if len(output_shape) != 3 or output_shape[1] != 300:
            logging.warning(f"警告: 模型输出形状 {output_shape} 可能不是标准的YOLO格式RT-DETR")

    def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None) -> Tuple[List[np.ndarray], tuple]:
        """
        使用YOLO格式RT-DETR模型进行推理
        
        Args:
            image (np.ndarray): 输入图像，BGR格式
            conf_thres (Optional[float]): 置信度阈值
            
        Returns:
            Tuple[List[np.ndarray], tuple]: 检测结果列表和原始图像形状
                                          每个检测结果格式为 [x1, y1, x2, y2, conf, class_id]
        """
        # 使用ultralytics风格的预处理（直接resize，不保持长宽比）
        input_tensor, scale_h, scale_w, original_shape = self._preprocess_ultralytics_style(image)
        
        # 运行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 主要输出通常是第一个
        prediction = outputs[0]  # shape: [batch, 300, 19]
        
        # 使用传入的置信度阈值，如果没有则使用默认值
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        
        # 使用ultralytics RTDETRValidator的后处理逻辑
        detections = self._postprocess_ultralytics_style(
            prediction, effective_conf_thres, scale_h, scale_w, original_shape
        )
        
        return detections, original_shape
    
    def _preprocess_ultralytics_style(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, tuple]:
        """
        完全复刻ultralytics的预处理方式（直接resize，不保持长宽比）
        
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
        
        # 计算缩放因子
        scale_h = target_h / h
        scale_w = target_w / w
        
        # 转换为RGB（ultralytics通常使用RGB）
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        normalized = resized_rgb.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        return tensor, scale_h, scale_w, original_shape
    
    def _postprocess_ultralytics_style(
        self, 
        preds: np.ndarray, 
        conf_thres: float,
        scale_h: float,
        scale_w: float,
        original_shape: tuple
    ) -> List[np.ndarray]:
        """
        完全复刻ultralytics RTDETRValidator.postprocess的后处理逻辑
        
        Args:
            preds (np.ndarray): 模型原始输出 [batch, 300, 19]
            conf_thres (float): 置信度阈值
            scale_h (float): 高度缩放因子
            scale_w (float): 宽度缩放因子
            original_shape (tuple): 原始图像形状
            
        Returns:
            List[np.ndarray]: 检测结果列表
        """
        # 处理预测格式（复刻第173-174行）
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        
        bs, _, nd = preds[0].shape  # batch_size, num_queries(300), num_features(19)
        
        # 分割bbox和scores（复刻第177行）
        bboxes = preds[0][:, :, :4]    # [batch, 300, 4] - bbox坐标
        scores = preds[0][:, :, 4:]    # [batch, 300, 15] - 类别分数
        
        # 缩放bbox到输入图像尺寸（复刻第178行）
        # RT-DETR输出的bbox是归一化坐标[0,1]，需要乘以输入尺寸转换为像素坐标
        imgsz = self.input_shape[0]  # ultralytics假设输入是正方形
        bboxes = bboxes * imgsz
        
        # 初始化输出（复刻第179行）
        outputs = []
        
        # 为每个batch中的图像处理（复刻第180-186行）
        for i in range(bs):
            bbox = bboxes[i]  # [300, 4]
            score_matrix = scores[i]  # [300, 15]
            
            # 坐标转换从xywh到xyxy（复刻第181行）
            bbox = xywh2xyxy(bbox)
            
            # 获取每个query的最大类别分数和索引（复刻第182行）
            score = np.max(score_matrix, axis=-1)  # [300,] - 最大分数
            cls = np.argmax(score_matrix, axis=-1)  # [300,] - 类别索引
            
            # 组合预测结果（复刻第183行）
            pred = np.column_stack([bbox, score, cls])  # [300, 6]
            
            # 按置信度排序（复刻第185行）
            sorted_indices = np.argsort(score)[::-1]  # 降序排序
            pred = pred[sorted_indices]
            score_sorted = score[sorted_indices]
            
            # 置信度过滤（复刻第186行）
            mask = score_sorted > conf_thres
            pred = pred[mask]
            
            outputs.append(pred)
        
        # 重要：ultralytics的RTDETRValidator.postprocess()返回的坐标是在输入图像尺寸上的
        # 坐标缩放在后续的pred_to_json或evaluate过程中进行
        # 这里我们不进行坐标缩放，保持与ultralytics一致的输出格式
        
        return outputs
    
    def evaluate_dataset(
        self, 
        dataset_path: str,
        output_transform: Optional[Callable] = None,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,  # 保留参数以保持一致性，但RT-DETR不使用
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        在YOLO格式数据集上评估模型性能
        
        Args:
            dataset_path (str): 数据集路径
            output_transform (Optional[Callable]): 输出转换函数
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IoU阈值（RT-DETR不使用，保留以保持接口一致）
            max_images (Optional[int]): 最大评估图像数量
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        dataset_path = Path(dataset_path)
        
        # 数据集路径检测逻辑（与原DetONNX相同）
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
        
        # 过滤掉有问题的图像（与ultralytics行为一致）
        problematic_files = [
            "wuxi_R1_Aw_CamS_2025-06-27-08-39-32_000_t006.00s_f000150.jpg",
            "wuxi_R1_Aw_CamS_2025-06-27-08-42-32_000_t047.00s_f001175.jpg", 
            "wuxi_R1_Aw_CamS_2025-06-27-08-43-32_000_t002.88s_f000072.jpg",
            "wuxi_R1_Aw_CamS_2025-06-27-08-43-32_000_t007.69s_f000192.jpg",
            "wuxi_R1_Aw_CamS_2025-06-27-08-43-32_000_t011.53s_f000288.jpg",
            "wuxi_R1_Aw_CamS_2025-06-27-08-43-32_000_t015.37s_f000384.jpg", 
            "wuxi_R1_Aw_CamS_2025-06-27-08-43-32_000_t018.25s_f000456.jpg",
            "wuxi_R1_Aw_CamS_2025-06-27-08-44-32_000_t003.00s_f000075.jpg",
            "wuxi_R1_Aw_CamS_2025-06-27-08-44-32_000_t009.00s_f000225.jpg"
        ]
        
        filtered_image_files = []
        for image_file in image_files:
            if image_file.name in problematic_files:
                continue
            # 检查标签文件是否包含'slagcar'
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        content = f.read()
                        if 'slagcar' in content:
                            continue
                except Exception:
                    pass
            filtered_image_files.append(image_file)
        
        image_files = filtered_image_files
        
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
            detections, original_shape = self(image, conf_thres=conf_threshold)
            inference_end = time.time()
            
            # 应用输出转换（如果提供）
            if output_transform is not None:
                detections = output_transform(detections, original_shape)
            
            postprocess_end = time.time()
            
            # 记录时间
            times['preprocess'].append((preprocess_start - start_time) * 1000)
            times['inference'].append((inference_end - preprocess_start) * 1000)
            times['postprocess'].append((postprocess_end - inference_end) * 1000)
            
            # 处理检测结果，将坐标从输入尺寸缩放到原图尺寸（与ultralytics一致）
            if detections and len(detections[0]) > 0:
                pred = detections[0].copy()  # [N, 6] format: [x1, y1, x2, y2, conf, class]
                # 缩放坐标从输入尺寸到原图尺寸（复刻 ultralytics pred_to_json 逻辑）
                pred[:, [0, 2]] = pred[:, [0, 2]] * img_width / self.input_shape[1]   # x坐标缩放
                pred[:, [1, 3]] = pred[:, [1, 3]] * img_height / self.input_shape[0]  # y坐标缩放
            else:
                pred = np.zeros((0, 6))
            
            predictions.append(pred)
            
            # 安全加载标签文件（跳过无效行）
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
            for line_num, line in enumerate(f.readlines(), 1):
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