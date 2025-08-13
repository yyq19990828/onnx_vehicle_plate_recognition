"""
目标检测指标计算模块

此模块提供目标检测的标准评估指标计算功能，包括：
- mAP (mean Average Precision)
- Precision & Recall
- F1-Score 
- IoU (Intersection over Union)

支持YOLO格式的标注和预测结果
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import yaml
import cv2
import os


def bbox_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    计算两组边界框之间的IoU
    
    Args:
        box1 (np.ndarray): 形状为 (N, 4) 的数组，格式为 [x1, y1, x2, y2]
        box2 (np.ndarray): 形状为 (M, 4) 的数组，格式为 [x1, y1, x2, y2]
        eps (float): 防止除零的小值
        
    Returns:
        np.ndarray: 形状为 (N, M) 的IoU矩阵
    """
    # 获取边界框坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # 计算交集区域
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # 计算并集区域
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = box1_area[:, None] + box2_area - inter_area

    # 计算IoU
    return inter_area / (union_area + eps)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    计算平均精度 (AP)
    
    Args:
        recall (np.ndarray): 召回率数组
        precision (np.ndarray): 精度数组
        
    Returns:
        float: 平均精度
    """
    # 在开头和结尾添加哨兵值
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 计算精度的单调递减序列
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 查找召回率发生变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算曲线下面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray, 
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个类别的AP、精度和召回率
    
    Args:
        tp (np.ndarray): 真正例数组
        conf (np.ndarray): 置信度数组
        pred_cls (np.ndarray): 预测类别数组
        target_cls (np.ndarray): 真实类别数组
        eps (float): 防止除零的小值
        
    Returns:
        Tuple: (ap, p, r, f1) - 每个类别的AP、精度、召回率和F1分数
    """
    # 按置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 获取唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别数量

    # 初始化输出数组
    ap = np.zeros((nc, tp.shape[1]))  # AP for each IoU threshold
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 该类别的标签数量
        n_p = i.sum()  # 该类别的预测数量

        if n_p == 0 or n_l == 0:
            continue

        # 累积FP和TP
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 召回率
        recall = tpc / (n_l + eps)

        # 负样本率: FP / (FP + TN) (可选)
        # fpr = fpc / (fpc[-1] + eps)

        # 精度
        precision = tpc / (tpc + fpc)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

        # 保存精度-召回率曲线 (用于绘图)
        py = np.interp(np.linspace(0, 1, 1000), recall[:, 0], precision[:, 0])  # 使用IoU=0.5
        p[ci] = py
        r[ci] = np.linspace(0, 1, 1000)

    # 计算F1分数
    f1 = 2 * p * r / (p + r + eps)

    return ap, p, r, f1


def process_batch(detections: np.ndarray, labels: np.ndarray, iou_thresholds: np.ndarray) -> np.ndarray:
    """
    处理一个批次的检测结果
    
    Args:
        detections (np.ndarray): 检测结果，形状为 (N, 6)，格式为 [x1, y1, x2, y2, conf, class]
        labels (np.ndarray): 真实标签，形状为 (M, 5)，格式为 [class, x1, y1, x2, y2]
        iou_thresholds (np.ndarray): IoU阈值数组
        
    Returns:
        np.ndarray: 形状为 (N, len(iou_thresholds)) 的TP矩阵
    """
    if detections.shape[0] == 0:
        return np.zeros((0, len(iou_thresholds)), dtype=bool)
    
    if labels.shape[0] == 0:
        return np.zeros((detections.shape[0], len(iou_thresholds)), dtype=bool)

    # 提取边界框和类别
    detection_boxes = detections[:, :4]
    detection_classes = detections[:, 5]
    
    label_boxes = labels[:, 1:5] 
    label_classes = labels[:, 0]

    # 计算IoU矩阵
    ious = bbox_iou(detection_boxes, label_boxes)

    # 初始化TP矩阵
    correct = np.zeros((detections.shape[0], len(iou_thresholds)), dtype=bool)

    # 对每个IoU阈值处理
    for i, iou_thresh in enumerate(iou_thresholds):
        # 匹配检测和标签
        x = np.where((ious >= iou_thresh) & (detection_classes[:, None] == label_classes))
        
        if x[0].shape[0] > 0:
            matches = np.concatenate((np.stack(x, 1), ious[x[0], x[1]][:, None]), 1)
            
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            
            correct[matches[:, 0].astype(int), i] = True

    return correct


class DetectionMetrics:
    """目标检测指标计算类"""
    
    def __init__(self, names: Dict[int, str] = None):
        """
        初始化检测指标计算器
        
        Args:
            names (Dict[int, str], optional): 类别名称字典
        """
        self.names = names or {}
        self.stats = {
            'tp': [],      # 真正例
            'conf': [],    # 置信度
            'pred_cls': [], # 预测类别
            'target_cls': [], # 真实类别
        }
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # IoU阈值 0.5:0.05:0.95
        
        # 额外统计信息
        self.class_stats = {}  # 每个类别的统计信息
        self.total_images = 0
        self.total_instances = 0
        
    def update_stats(self, tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray, gt_boxes: np.ndarray = None):
        """
        更新统计信息
        
        Args:
            tp (np.ndarray): 真正例数组
            conf (np.ndarray): 置信度数组  
            pred_cls (np.ndarray): 预测类别数组
            target_cls (np.ndarray): 真实类别数组
            gt_boxes (np.ndarray, optional): 真实标签框，用于计算尺寸统计
        """
        self.stats['tp'].append(tp)
        self.stats['conf'].append(conf)
        self.stats['pred_cls'].append(pred_cls)
        self.stats['target_cls'].append(target_cls)
        
        # 更新图像和实例计数
        self.total_images += 1
        if len(target_cls) > 0:
            self.total_instances += len(target_cls)
            
            # 统计每个类别的信息
            unique_classes = np.unique(target_cls)
            for cls in unique_classes:
                cls_int = int(cls)
                if cls_int not in self.class_stats:
                    self.class_stats[cls_int] = {
                        'images': set(),
                        'instances': 0,
                        'boxes_small': 0,
                        'boxes_medium': 0, 
                        'boxes_large': 0
                    }
                
                self.class_stats[cls_int]['images'].add(self.total_images - 1)
                cls_mask = target_cls == cls
                self.class_stats[cls_int]['instances'] += np.sum(cls_mask)
                
                # 如果有边界框信息，计算尺寸分布
                if gt_boxes is not None and len(gt_boxes) > 0:
                    cls_boxes = gt_boxes[cls_mask]
                    if len(cls_boxes) > 0:
                        # 计算面积 (x2-x1) * (y2-y1)
                        areas = (cls_boxes[:, 3] - cls_boxes[:, 1]) * (cls_boxes[:, 4] - cls_boxes[:, 2])
                        
                        # COCO标准的尺寸划分：small < 32²，medium < 96²，large >= 96²
                        small_mask = areas < 32*32
                        medium_mask = (areas >= 32*32) & (areas < 96*96)
                        large_mask = areas >= 96*96
                        
                        self.class_stats[cls_int]['boxes_small'] += np.sum(small_mask)
                        self.class_stats[cls_int]['boxes_medium'] += np.sum(medium_mask)
                        self.class_stats[cls_int]['boxes_large'] += np.sum(large_mask)
    
    def process(self) -> Dict[str, Any]:
        """
        处理所有统计信息并计算最终指标
        
        Returns:
            Dict[str, Any]: 包含所有指标的字典
        """
        # 检查是否有数据
        if not any(len(v) > 0 for v in self.stats.values()):
            return {}
        
        # 合并所有批次的统计信息，处理空列表情况
        stats = {}
        for k, v in self.stats.items():
            if len(v) > 0:
                stats[k] = np.concatenate(v, 0)
            else:
                stats[k] = np.array([])
        
        if len(stats.get('tp', [])) == 0:
            return {}
        
        # 计算每个类别的AP
        ap, p, r, f1 = ap_per_class(
            stats['tp'],
            stats['conf'], 
            stats['pred_cls'],
            stats['target_cls']
        )
        
        # 获取唯一类别
        unique_classes = np.unique(np.concatenate(self.stats['target_cls'])) if self.stats['target_cls'] else []
        
        # 计算mAP
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        
        # 构建每个类别的详细统计
        class_images = {}
        class_instances = {}
        class_precision = {}
        class_recall = {}
        
        for i, cls in enumerate(unique_classes):
            cls_int = int(cls)
            if cls_int in self.class_stats:
                class_images[cls_int] = len(self.class_stats[cls_int]['images'])
                class_instances[cls_int] = self.class_stats[cls_int]['instances']
                
                # 使用计算出的精度和召回率
                if i < len(p) and len(p[i]) > 0:
                    class_precision[cls_int] = p[i].mean()
                    class_recall[cls_int] = r[i].mean()
                else:
                    class_precision[cls_int] = 0.0
                    class_recall[cls_int] = 0.0
        
        # 构建结果字典
        results = {
            'map': map,           # mAP@0.5:0.95
            'map50': map50,       # mAP@0.5
            'mp': mp,             # 平均精度
            'mr': mr,             # 平均召回率
            'ap': ap,             # 每个类别的AP
            'ap50': ap50,         # 每个类别的AP@0.5
            'p': p,               # 精度曲线
            'r': r,               # 召回率曲线
            'f1': f1,             # F1分数
            
            # 额外的统计信息
            'total_images': self.total_images,
            'total_instances': self.total_instances,
            'class_images': class_images,
            'class_instances': class_instances,
            'class_precision': class_precision,
            'class_recall': class_recall,
        }
        
        # 计算按尺寸划分的指标（如果有尺寸信息）
        if any('boxes_small' in stats for stats in self.class_stats.values()):
            # 这里可以添加按尺寸计算AP的逻辑
            # 暂时作为占位符
            results['map50_small'] = 0.0
            results['map_small'] = 0.0
            results['map50_medium'] = 0.0
            results['map_medium'] = 0.0
            results['map50_large'] = 0.0
            results['map_large'] = 0.0
        
        return results


def load_yolo_labels(label_path: str, img_width: int, img_height: int) -> np.ndarray:
    """
    加载YOLO格式的标签文件
    
    Args:
        label_path (str): 标签文件路径
        img_width (int): 图像宽度
        img_height (int): 图像高度
        
    Returns:
        np.ndarray: 标签数组，格式为 [class, x1, y1, x2, y2]
    """
    if not os.path.exists(label_path):
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
                except ValueError as e:
                    # print(f"警告: 跳过标签文件 {label_path} 第 {line_num} 行，遇到无效字符: {parts[0]} - {e}")
                    logging.warning(f"跳过标签文件 {label_path} 第 {line_num} 行，遇到无效字符: {parts[0]}")
                    continue
    
    return np.array(labels) if labels else np.zeros((0, 5))


def evaluate_detection(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray], 
    names: Dict[int, str] = None
) -> Dict[str, Any]:
    """
    评估目标检测结果
    
    Args:
        predictions (List[np.ndarray]): 预测结果列表，每个元素格式为 [x1, y1, x2, y2, conf, class]
        ground_truths (List[np.ndarray]): 真实标签列表，每个元素格式为 [class, x1, y1, x2, y2]
        names (Dict[int, str], optional): 类别名称字典
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    metrics = DetectionMetrics(names)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # 收集所有在真实标签中出现的类别
    all_gt_classes = set()
    for gt in ground_truths:
        if len(gt) > 0:
            all_gt_classes.update(gt[:, 0].astype(int))
    all_gt_classes = sorted(list(all_gt_classes))
    
    # 处理每个图像的预测和标签
    for pred, gt in zip(predictions, ground_truths):
        if len(pred) == 0:
            # 没有预测结果
            if len(gt) > 0:
                # 但有真实标签，记录为未检测到的目标
                # 为了保持数据一致性，添加空的统计信息
                metrics.stats['target_cls'].append(gt[:, 0])
                metrics.stats['tp'].append(np.array([]).reshape(0, len(iou_thresholds)))
                metrics.stats['conf'].append(np.array([]))
                metrics.stats['pred_cls'].append(np.array([]))
            continue
            
        if len(gt) == 0:
            # 没有真实标签，所有预测都是假正例
            tp = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)
            metrics.update_stats(tp, pred[:, 4], pred[:, 5], np.array([]), None)
            continue
        
        # 处理有预测和标签的情况
        tp = process_batch(pred, gt, iou_thresholds)
        metrics.update_stats(tp, pred[:, 4], pred[:, 5], gt[:, 0], gt)
    
    results = metrics.process()
    
    # 添加类别信息，确保包含所有在真实标签中出现的类别
    if results and 'ap50' in results:
        # 获取当前计算出的类别
        unique_classes, _ = np.unique(np.concatenate(metrics.stats['target_cls']) if metrics.stats['target_cls'] else [], return_counts=True)
        results['classes'] = unique_classes.astype(int).tolist()
        results['all_gt_classes'] = all_gt_classes
    
    return results


def print_metrics(results: Dict[str, Any], names: Dict[int, str] = None):
    """
    以Ultralytics风格打印指标结果
    
    Args:
        results (Dict[str, Any]): 评估结果
        names (Dict[int, str], optional): 类别名称字典
    """
    if not results:
        print("无有效检测结果")
        return
    
    # 获取统计信息
    total_images = results.get('total_images', 0)
    total_instances = results.get('total_instances', 0)
    
    print(f"\n                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):")
    
    # 打印每个类别的结果
    if 'ap50' in results and 'ap' in results:
        computed_classes = results.get('classes', [])
        all_gt_classes = results.get('all_gt_classes', computed_classes)
        class_to_idx = {cls: i for i, cls in enumerate(computed_classes)}
        
        # 获取每个类别的图像数和实例数
        class_images = results.get('class_images', {})
        class_instances = results.get('class_instances', {})
        class_precision = results.get('class_precision', {})
        class_recall = results.get('class_recall', {})
        
        # 按类别索引排序显示
        for class_id in sorted(all_gt_classes):
            class_name = names.get(class_id, f"class{class_id}") if names else f"class{class_id}"
            
            # 限制类别名称长度
            if len(class_name) > 16:
                class_name = class_name[:13] + "..."
            
            if class_id in class_to_idx:
                idx = class_to_idx[class_id]
                ap50 = results['ap50'][idx]
                ap = results['ap'][idx]
                precision = class_precision.get(class_id, 0)
                recall = class_recall.get(class_id, 0)
                images = class_images.get(class_id, 0)
                instances = class_instances.get(class_id, 0)
                
                print(f"               {class_name:>12s}      {images:4d}      {instances:4d}      {precision:.3f}      {recall:.3f}      {ap50:.3f}      {ap:.3f}")
            else:
                # 没有预测结果的类别
                images = class_images.get(class_id, 0)
                instances = class_instances.get(class_id, 0)
                print(f"               {class_name:>12s}      {images:4d}      {instances:4d}          0          0          0          0")
    
    # 打印总体指标
    map50 = results.get('map50', 0)
    map50_95 = results.get('map', 0)
    mp = results.get('mp', 0)
    mr = results.get('mr', 0)
    
    print(f"                   all      {total_images:4d}     {total_instances:5d}      {mp:.3f}      {mr:.3f}      {map50:.3f}      {map50_95:.3f}")
    
    # 添加按尺寸划分的指标（如果可用）
    if 'map50_small' in results or 'map50_medium' in results or 'map50_large' in results:
        print(f"\n                             mAP50   mAP50-95")
        if 'map50_small' in results:
            print(f"               small:      {results.get('map50_small', 0):.3f}      {results.get('map_small', 0):.3f}")
        if 'map50_medium' in results:
            print(f"               medium:     {results.get('map50_medium', 0):.3f}      {results.get('map_medium', 0):.3f}")
        if 'map50_large' in results:
            print(f"               large:      {results.get('map50_large', 0):.3f}      {results.get('map_large', 0):.3f}")
    
    # 打印速度统计（如果可用）
    if 'speed_preprocess' in results:
        speed_preprocess = results.get('speed_preprocess', 0)
        speed_inference = results.get('speed_inference', 0) 
        speed_loss = results.get('speed_loss', 0)
        speed_postprocess = results.get('speed_postprocess', 0)
        
        print(f"Speed: {speed_preprocess:.1f}ms preprocess, {speed_inference:.1f}ms inference, {speed_loss:.1f}ms loss, {speed_postprocess:.1f}ms postprocess per image")