import numpy as np

def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def sigmoid(x):
    """Compute sigmoid values for each sets of scores in x."""
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x_center, y_center, width, height) to (x1, y1, x2, y2).

    Args:
        x (np.ndarray): Bounding box coordinates in xywh format.

    Returns:
        np.ndarray: Bounding box coordinates in xyxy format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.5,
    iou_thres: float = 0.5,
    classes: list = None,
    agnostic: bool = False,
    multi_label: bool = True,  # 默认值改为True，与Ultralytics一致
    max_det: int = 300,
    model_type: str = "yolo",  # 新增参数：模型类型
    has_objectness: bool = False,  # 新增参数：是否有objectness分支，默认False
) -> list:
    """
    Perform Non-Maximum Suppression (NMS) on inference results.
    
    与Ultralytics对齐的实现，支持现代YOLO格式：
    - 有objectness: [batch, num_anchors, 4 + 1 + num_classes] (传统YOLO)
    - 无objectness: [batch, num_anchors, 4 + num_classes] (现代YOLO如YOLO11)

    Args:
        prediction (np.ndarray): 模型原始输出，形状为[batch, num_anchors, features]
        conf_thres (float): 置信度阈值
        iou_thres (float): IoU阈值
        classes (list, optional): 要考虑的类别索引列表
        agnostic (bool): 是否执行类别无关的NMS
        multi_label (bool): 是否考虑每个框的多个标签
        max_det (int): 保留的最大检测数量
        model_type (str): 模型类型，默认"yolo"
        has_objectness (bool): 是否有objectness分支，默认False（适应现代YOLO）

    Returns:
        list: 每个图像的检测结果列表，每个检测为[x1, y1, x2, y2, conf, class_id]格式
    """
    # Settings
    max_nms = 30000  # maximum number of boxes into NMS

    bs = prediction.shape[0]  # batch size
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # YOLO格式处理：根据has_objectness参数决定如何解析输出
        # 传统YOLO: [num_anchors, 4 + 1 + num_classes] - 有objectness
        # 现代YOLO: [num_anchors, 4 + num_classes] - 无objectness
        
        if has_objectness:  # 传统YOLO格式，有objectness分支
            # 提取objectness和类别分数
            obj_conf = x[:, 4:5]  # objectness score
            cls_conf = x[:, 5:]   # class scores
            
            # 处理原始logits（如果需要）
            if np.max(obj_conf) > 1:
                obj_conf = sigmoid(obj_conf)
            if np.max(cls_conf) > 1:
                cls_conf = sigmoid(cls_conf)
            
            if multi_label:
                # multi_label模式：每个框可以有多个类别
                # 使用obj_conf * cls_conf作为最终置信度
                class_scores = obj_conf * cls_conf
                # 在multi_label模式下，需要处理多个类别
                # 这里暂时简化为取最大值（后续可以优化）
                conf = np.max(class_scores, axis=1, keepdims=True)
                mask = (conf.flatten() >= conf_thres)
                
                if not np.any(mask):
                    continue
                    
                x = x[mask]
                class_scores = class_scores[mask]
                conf = conf[mask]
                j = np.argmax(class_scores, axis=1, keepdims=True)
            else:
                # 单标签模式：每个框只有一个类别
                # 使用obj_conf * max(cls_conf)作为置信度
                class_scores = cls_conf
                conf = obj_conf.flatten() * np.max(class_scores, axis=1)
                mask = (conf >= conf_thres)
                
                if not np.any(mask):
                    continue
                    
                x = x[mask]
                class_scores = class_scores[mask]
                conf = conf[mask].reshape(-1, 1)
                
                # 获取最大类别索引
                j = np.argmax(class_scores, axis=1, keepdims=True)
        
        else:  # 现代YOLO格式（如YOLO11）：没有单独的objectness
            # 直接使用类别分数
            class_scores = x[:, 4:]
            
            # 处理原始logits（如果需要）
            if np.max(class_scores) > 1:
                class_scores = sigmoid(class_scores)
            
            # 计算置信度和类别
            conf = np.max(class_scores, axis=1, keepdims=True)
            mask = (conf.flatten() >= conf_thres)
            
            if not np.any(mask):
                continue
                
            x = x[mask]
            conf = conf[mask]
            class_scores = class_scores[mask]
            j = np.argmax(class_scores, axis=1, keepdims=True)

        # Convert box from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Create the detections matrix [x1, y1, x2, y2, conf, class_id]
        x = np.concatenate((box, conf, j.astype(np.float32)), 1)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

        # Batched NMS
        max_wh = 7680  # maximum box width and height
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # NMS using pure numpy (修正IoU计算，移除+1偏移)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            # 修正：移除+1偏移，与Ultralytics保持一致
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算IoU（不使用+1偏移）
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area_i + area_order - inter
            # 避免除零
            union = np.maximum(union, 1e-6)
            ovr = inter / union

            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]

        i = np.array(keep)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output