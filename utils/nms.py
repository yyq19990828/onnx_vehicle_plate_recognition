import numpy as np

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
    multi_label: bool = False,
    max_det: int = 300,
) -> list:
    """
    Perform Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction (np.ndarray): The model's raw output, shape [batch, num_boxes, 5+num_classes].
        conf_thres (float): Confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): A list of class indices to consider. Defaults to None.
        agnostic (bool): If True, perform class-agnostic NMS. Defaults to False.
        multi_label (bool): If True, consider multiple labels per box. Defaults to False.
        max_det (int): Maximum number of detections to keep. Defaults to 300.

    Returns:
        list: A list of detections for each image in the batch.
              Each detection is a tensor of shape [num_detections, 6] -> (x1, y1, x2, y2, conf, class_id).
    """
    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5  # seconds to quit after

    # Based on user feedback, the model output is [xywh, class_scores...]
    # where class_scores are final confidences (obj_conf * class_conf).
    # The logic is simplified to handle this format directly.
    bs = prediction.shape[0]  # batch size
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # The model output is [xywh, class_scores...].
        # We need to find the class with the highest score for each box.
        
        # Filter out boxes where the max class score is below the confidence threshold.
        class_scores = x[:, 4:]
        conf = np.max(class_scores, axis=1, keepdims=True)
        
        mask = (conf.flatten() >= conf_thres)
        if not np.any(mask):
            continue
            
        x = x[mask]
        conf = conf[mask]
        
        # Get the corresponding class index for the filtered boxes
        j = np.argmax(x[:, 4:], axis=1, keepdims=True)

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
        
        # NMS using pure numpy
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / ((boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1) + (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (boxes[order[1:], 3] - boxes[order[1:], 1] + 1) - inter)

            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]

        i = np.array(keep)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output