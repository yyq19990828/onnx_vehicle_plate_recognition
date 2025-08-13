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
        
    def update_stats(self, tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray):
        """
        更新统计信息
        
        Args:
            tp (np.ndarray): 真正例数组
            conf (np.ndarray): 置信度数组  
            pred_cls (np.ndarray): 预测类别数组
            target_cls (np.ndarray): 真实类别数组
        """
        self.stats['tp'].append(tp)
        self.stats['conf'].append(conf)
        self.stats['pred_cls'].append(pred_cls)
        self.stats['target_cls'].append(target_cls)
    
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
        
        # 计算mAP
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        
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
        }
        
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
            metrics.update_stats(tp, pred[:, 4], pred[:, 5], np.array([]))
            continue
        
        # 处理有预测和标签的情况
        tp = process_batch(pred, gt, iou_thresholds)
        metrics.update_stats(tp, pred[:, 4], pred[:, 5], gt[:, 0])
    
    return metrics.process()


def print_metrics(results: Dict[str, Any], names: Dict[int, str] = None):
    """
    打印指标结果
    
    Args:
        results (Dict[str, Any]): 评估结果
        names (Dict[int, str], optional): 类别名称字典
    """
    if not results:
        print("无有效检测结果")
        return
    
    print(f"\n{'类别':<15} {'AP@0.5':<10} {'AP@0.5:0.95':<12}")
    print("-" * 40)
    
    # 打印每个类别的结果
    if 'ap50' in results and 'ap' in results:
        for i, (ap50, ap) in enumerate(zip(results['ap50'], results['ap'])):
            class_name = names.get(i, f"class_{i}") if names else f"class_{i}"
            print(f"{class_name:<15} {ap50:<10.3f} {ap:<12.3f}")
    
    print("-" * 40)
    print(f"{'总体指标':<15}")
    print(f"mAP@0.5      : {results.get('map50', 0):.3f}")
    print(f"mAP@0.5:0.95 : {results.get('map', 0):.3f}")
    print(f"精度 (P)     : {results.get('mp', 0):.3f}")
    print(f"召回率 (R)   : {results.get('mr', 0):.3f}")