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
) -> list:
    """
    Perform Non-Maximum Suppression (NMS) on inference results.
    
    与Ultralytics对齐的实现，正确处理YOLO格式：[batch, num_anchors, 4 + 1 + num_classes]

    Args:
        prediction (np.ndarray): The model's raw output.
                                For YOLO: shape [batch, num_anchors, 4 + 1 + num_classes]
                                where 4=bbox, 1=objectness, num_classes=class scores
        conf_thres (float): Confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): A list of class indices to consider. Defaults to None.
        agnostic (bool): If True, perform class-agnostic NMS. Defaults to False.
        multi_label (bool): If True, consider multiple labels per box. Defaults to True.
        max_det (int): Maximum number of detections to keep. Defaults to 300.
        model_type (str): Model type ("yolo" or "rtdetr"). Defaults to "yolo".

    Returns:
        list: A list of detections for each image in the batch.
              Each detection is a tensor of shape [num_detections, 6] -> (x1, y1, x2, y2, conf, class_id).
    """
    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5  # seconds to quit after

    bs = prediction.shape[0]  # batch size
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # YOLO格式: [num_anchors, 4 + 1 + num_classes]
        # 其中: x[:, :4] = bbox (xywh)
        #      x[:, 4] = objectness 
        #      x[:, 5:] = class scores
        
        if model_type == "yolo" and x.shape[1] > 5:  # 标准YOLO格式，有objectness
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
        
        else:  # 简化格式或RT-DETR格式：没有单独的objectness
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