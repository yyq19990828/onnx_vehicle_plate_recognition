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
            'target_img': [], # 目标图像索引（用于计算每类别在图像中的分布）
        }
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # IoU阈值 0.5:0.05:0.95
        
        # 额外统计信息
        self.class_stats = {}  # 每个类别的统计信息
        self.total_images = 0
        self.total_instances = 0
        self.seen = 0  # 已处理的图像数量
        
        # 结果缓存（类似ultralytics的DetMetrics）
        self.ap = []           # 每个类别的AP@0.5:0.95
        self.ap50 = []         # 每个类别的AP@0.5
        self.ap_class_index = []  # AP对应的类别索引
        self.p = []            # 每个类别的precision
        self.r = []            # 每个类别的recall
        self.f1 = []           # 每个类别的f1
        self.mp = 0.0          # 平均precision
        self.mr = 0.0          # 平均recall
        self.map50 = 0.0       # mAP@0.5
        self.map = 0.0         # mAP@0.5:0.95
        self.nt_per_class = None  # 每个类别的目标总数
        self.nt_per_image = None  # 每个类别在图像中的分布
        
    def update_stats(self, stats_dict: Dict[str, np.ndarray]):
        """
        更新统计信息（类似ultralytics DetMetrics.update_stats）
        
        Args:
            stats_dict (Dict[str, np.ndarray]): 包含统计信息的字典
                - tp: 真正例数组
                - conf: 置信度数组  
                - pred_cls: 预测类别数组
                - target_cls: 真实类别数组
                - target_img: 目标图像索引数组
        """
        for k in self.stats.keys():
            if k in stats_dict:
                self.stats[k].append(stats_dict[k])
        
        # 统计图像处理数量
        self.seen += 1
        
        # 更新图像和实例计数
        if len(stats_dict.get('target_cls', [])) > 0:
            target_cls = stats_dict['target_cls']
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
                
                self.class_stats[cls_int]['images'].add(self.seen - 1)
                cls_mask = target_cls == cls
                self.class_stats[cls_int]['instances'] += np.sum(cls_mask)
    
    def process(self) -> Dict[str, Any]:
        """
        处理所有统计信息并计算最终指标（类似ultralytics DetMetrics.process）
        
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
        unique_classes = np.unique(stats['target_cls']).astype(int) if len(stats['target_cls']) > 0 else []
        
        # 存储结果（类似ultralytics）
        self.ap = ap.mean(1) if len(ap) else []  # AP@0.5:0.95
        self.ap50 = ap[:, 0] if len(ap) else []  # AP@0.5
        self.ap_class_index = unique_classes
        
        # 计算每个类别在最大F1时的precision和recall
        if len(f1) > 0:
            f1_mean = f1.mean(0)
            max_f1_idx = f1_mean.argmax()
            self.p = p[:, max_f1_idx] if len(p) else []
            self.r = r[:, max_f1_idx] if len(r) else []
            self.f1 = f1[:, max_f1_idx] if len(f1) else []
        else:
            self.p = []
            self.r = []
            self.f1 = []
        
        # 计算总体指标
        self.mp = self.p.mean() if len(self.p) else 0.0
        self.mr = self.r.mean() if len(self.r) else 0.0
        self.map50 = self.ap50.mean() if len(self.ap50) else 0.0
        self.map = self.ap.mean() if len(self.ap) else 0.0
        
        # 计算每个类别的目标数统计（类似ultralytics）
        if len(stats['target_cls']) > 0:
            all_target_cls = stats['target_cls'].astype(int)
            self.nt_per_class = np.bincount(all_target_cls, minlength=len(self.names))
            
            # 计算每个类别在多少张图像中出现
            if len(stats.get('target_img', [])) > 0:
                # 为每个类别计算它们出现在多少张不同图像中
                self.nt_per_image = np.zeros(len(self.names), dtype=int)
                for cls_id in unique_classes:
                    if cls_id < len(self.nt_per_image):
                        # 找到该类别在哪些图像中出现
                        cls_mask = all_target_cls == cls_id
                        unique_images = np.unique(stats['target_img'][cls_mask])
                        self.nt_per_image[cls_id] = len(unique_images)
            else:
                # 如果没有target_img，使用类别统计信息估算
                self.nt_per_image = np.zeros(len(self.names), dtype=int)
                for cls_id in unique_classes:
                    if cls_id < len(self.nt_per_image) and cls_id in self.class_stats:
                        self.nt_per_image[cls_id] = len(self.class_stats[cls_id]['images'])
        else:
            self.nt_per_class = np.zeros(len(self.names), dtype=int)
            self.nt_per_image = np.zeros(len(self.names), dtype=int)
        
        # 构建结果字典
        results_dict = {
            'map': self.map,           # mAP@0.5:0.95
            'map50': self.map50,       # mAP@0.5
            'mp': self.mp,             # 平均精度
            'mr': self.mr,             # 平均召回率
            'ap': self.ap,             # 每个类别的AP
            'ap50': self.ap50,         # 每个类别的AP@0.5
            'p': p,                    # 精度曲线
            'r': r,                    # 召回率曲线
            'f1': f1,                  # F1分数
            'ap_class_index': self.ap_class_index,  # 类别索引
            
            # 额外的统计信息
            'total_images': self.seen,
            'total_instances': self.total_instances,
        }
        
        return results_dict
    
    def mean_results(self) -> List[float]:
        """返回平均结果：precision, recall, mAP50, mAP50-95"""
        return [self.mp, self.mr, self.map50, self.map]
    
    def class_result(self, i: int) -> Tuple[float, float, float, float]:
        """返回第i个类别的结果：precision, recall, AP50, AP50-95"""
        if i < len(self.p):
            return self.p[i], self.r[i], self.ap50[i], self.ap[i]
        return 0.0, 0.0, 0.0, 0.0
    
    @property
    def keys(self) -> List[str]:
        """返回指标键列表"""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    
    def get_desc(self) -> str:
        """返回格式化的类别指标总结字符串（类似DetectionValidator.get_desc）"""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
    
    def clear_stats(self):
        """清除存储的统计信息"""
        for v in self.stats.values():
            v.clear()
        self.class_stats.clear()
        self.total_images = 0
        self.total_instances = 0
        self.seen = 0


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
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # 准备统计字典
        stats_dict = {
            'target_cls': gt[:, 0] if len(gt) > 0 else np.array([]),
            'target_img': np.full(len(gt), i) if len(gt) > 0 else np.array([])  # 图像索引
        }
        
        if len(pred) == 0:
            # 没有预测结果
            if len(gt) > 0:
                # 但有真实标签，记录为未检测到的目标
                stats_dict.update({
                    'tp': np.array([]).reshape(0, len(iou_thresholds)),
                    'conf': np.array([]),
                    'pred_cls': np.array([])
                })
            else:
                # 没有预测也没有标签，跳过
                continue
        else:
            if len(gt) == 0:
                # 没有真实标签，所有预测都是假正例
                tp = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)
                stats_dict.update({
                    'tp': tp,
                    'conf': pred[:, 4],
                    'pred_cls': pred[:, 5]
                })
            else:
                # 处理有预测和标签的情况
                tp = process_batch(pred, gt, iou_thresholds)
                stats_dict.update({
                    'tp': tp,
                    'conf': pred[:, 4],
                    'pred_cls': pred[:, 5]
                })
        
        # 更新统计信息
        metrics.update_stats(stats_dict)
    
    # 处理结果
    results = metrics.process()
    
    # 添加兼容信息
    if results:
        results['classes'] = metrics.ap_class_index
        results['all_gt_classes'] = all_gt_classes
    
    return results


def print_results(metrics: 'DetectionMetrics', names: Dict[int, str] = None, training: bool = False):
    """
    以Ultralytics DetectionValidator风格打印训练/验证集每个类别的指标
    
    Args:
        metrics (DetectionMetrics): 计算好的检测指标
        names (Dict[int, str], optional): 类别名称字典
        training (bool): 是否为训练模式
    """
    # 使用与DetectionValidator.get_desc()相同的格式
    pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)  # print format
    
    # 总体指标（类似DetectionValidator.print_results中的总体打印）
    total_seen = metrics.seen
    total_targets = metrics.nt_per_class.sum() if metrics.nt_per_class is not None else 0
    
    # 打印总体结果
    print(pf % ("all", total_seen, total_targets, *metrics.mean_results()))
    
    # 检查是否有标签
    if total_targets == 0:
        print(f"WARNING ⚠️ no labels found in {'train' if training else 'val'} set, can not compute metrics without labels")
        return
    
    # 打印每个类别的结果
    nc = len(names) if names else max(metrics.ap_class_index) + 1 if len(metrics.ap_class_index) > 0 else 0
    if not training and nc > 1 and len(metrics.ap_class_index) > 0:
        for i, c in enumerate(metrics.ap_class_index):
            class_name = names.get(c, f"class{c}") if names else f"class{c}"
            
            # 获取该类别的图像数和实例数
            images_with_class = metrics.nt_per_image[c] if c < len(metrics.nt_per_image) else 0
            instances_of_class = metrics.nt_per_class[c] if c < len(metrics.nt_per_class) else 0
            
            print(pf % (
                class_name,
                images_with_class,
                instances_of_class,
                *metrics.class_result(i),
            ))


def print_metrics(results: Dict[str, Any], names: Dict[int, str] = None):
    """
    以Ultralytics风格打印指标结果（保留原有接口兼容性）
    
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
    
    # 打印表头（类似DetectionValidator.get_desc()）
    header = f"{'Class':>22} {'Images':>11} {'Instances':>11} {'Box(P':>11} {'R':>11} {'mAP50':>11} {'mAP50-95)':>11}"
    print(header)
    
    # 打印每个类别的结果
    if 'ap50' in results and 'ap' in results:
        computed_classes = results.get('classes', [])
        all_gt_classes = results.get('all_gt_classes', computed_classes)
        class_to_idx = {cls: i for i, cls in enumerate(computed_classes)}
        
        # 按类别索引排序显示
        for class_id in sorted(all_gt_classes):
            class_name = names.get(class_id, f"class{class_id}") if names else f"class{class_id}"
            
            # 限制类别名称长度
            if len(class_name) > 22:
                class_name = class_name[:19] + "..."
            
            if class_id in class_to_idx:
                idx = class_to_idx[class_id]
                ap50 = results['ap50'][idx]
                ap = results['ap'][idx]
                
                # 尝试从结果中获取精度和召回率
                if 'p' in results and len(results['p']) > idx:
                    if hasattr(results['p'][idx], 'mean'):
                        precision = results['p'][idx].mean()
                    else:
                        precision = results['p'][idx] if np.isscalar(results['p'][idx]) else 0
                else:
                    precision = 0
                
                if 'r' in results and len(results['r']) > idx:
                    if hasattr(results['r'][idx], 'mean'):
                        recall = results['r'][idx].mean()
                    else:
                        recall = results['r'][idx] if np.isscalar(results['r'][idx]) else 0
                else:
                    recall = 0
                
                # 图像数和实例数（简化处理）
                images = 1  # 占位符
                instances = 1  # 占位符
                
                print(f"{class_name:>22} {images:>11d} {instances:>11d} {precision:>11.3g} {recall:>11.3g} {ap50:>11.3g} {ap:>11.3g}")
            else:
                # 没有预测结果的类别
                print(f"{class_name:>22} {0:>11d} {0:>11d} {0:>11.3g} {0:>11.3g} {0:>11.3g} {0:>11.3g}")

    # 打印总体指标
    map50 = results.get('map50', 0)
    map50_95 = results.get('map', 0)
    mp = results.get('mp', 0)
    mr = results.get('mr', 0)
    
    print(f"{'all':>22} {total_images:>11d} {total_instances:>11d} {mp:>11.3g} {mr:>11.3g} {map50:>11.3g} {map50_95:>11.3g}")
    
    # 打印速度统计（如果可用）
    if 'speed_preprocess' in results:
        speed_preprocess = results.get('speed_preprocess', 0)
        speed_inference = results.get('speed_inference', 0) 
        speed_loss = results.get('speed_loss', 0)
        speed_postprocess = results.get('speed_postprocess', 0)
        
        print(f"Speed: {speed_preprocess:.1f}ms preprocess, {speed_inference:.1f}ms inference, {speed_loss:.1f}ms loss, {speed_postprocess:.1f}ms postprocess per image")