import cv2
import numpy as np

def detect_skew_angle(image):
    """检测图像的倾斜角度，输入灰度图像，返回角度（度）"""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return 0
    return np.median(angles)

def correct_skew(image, angle):
    """校正图像倾斜，输入BGR图像和角度，返回校正后图像"""
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return corrected

def find_optimal_split_line(gray_img):
    """通过水平投影找到最佳分割线位置，输入灰度图像，返回y坐标"""
    height, width = gray_img.shape
    search_start = int(height * 0.25)
    search_end = int(height * 0.65)
    horizontal_projection = np.sum(gray_img, axis=1)
    kernel_size = max(3, height // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed_projection = cv2.GaussianBlur(horizontal_projection.astype(np.float32).reshape(-1, 1), (1, kernel_size), 0).flatten()
    search_region = smoothed_projection[search_start:search_end]
    if len(search_region) == 0:
        return int(height * 0.35)
    max_val = np.max(search_region)
    max_positions = []
    threshold = max_val * 0.9
    for i in range(len(search_region)):
        if search_region[i] >= threshold:
            max_positions.append(search_start + i)
    if max_positions:
        split_point = max_positions[len(max_positions) // 2]
        return split_point
    projection_diff = np.abs(np.diff(smoothed_projection[search_start:search_end]))
    if len(projection_diff) > 0:
        max_change_idx = np.argmax(projection_diff)
        return search_start + max_change_idx
    return int(height * 0.35)

def process_plate_image(img, is_double_layer=False, verbose=False):
    """
    处理车牌图像（Numpy数组），如为双层则自动校正、分割、拼接为单层
    输入: img (BGR), is_double_layer (bool)
    返回: 单层车牌图像 (BGR)
    """
    if img is None or img.size == 0:
        if verbose:
            print("输入图像为空")
        return None
    # 灰度
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 倾斜校正
    skew_angle = detect_skew_angle(gray_img)
    corrected_img = correct_skew(img, skew_angle)
    if not is_double_layer:
        return corrected_img
    # 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray_img = clahe.apply(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
    # 分割
    split_point = find_optimal_split_line(enhanced_gray_img)
    top_part = corrected_img[0:split_point, :]
    bottom_part = corrected_img[split_point:, :]
    bottom_h, bottom_w = bottom_part.shape[:2]
    top_h, top_w = top_part.shape[:2]
    if bottom_h <= 0 or bottom_w <= 0 or top_h <= 0 or top_w <= 0:
        if verbose:
            print("分割后部分为空")
        return None
    # 上层缩放到下层高度并收窄
    target_height = bottom_h
    top_aspect_ratio = top_w / top_h
    target_top_width = int(target_height * top_aspect_ratio * 0.5)
    top_resized = cv2.resize(top_part, (target_top_width, target_height), interpolation=cv2.INTER_LINEAR)
    stitched_plate = cv2.hconcat([top_resized, bottom_part])
    return stitched_plate

# 预处理函数将在后续补充
def image_pretreatment(img, default_size=(168, 48)):
    """图像处理成固定的尺寸，归一化，适用于颜色/层数模型（无torch依赖）"""
    img = np.asarray(img).astype(np.float32)
    img = cv2.resize(img, default_size)
    mean_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    onnxInferData = img[np.newaxis, :, :, :]
    onnxInferData = np.array(onnxInferData, dtype=np.float32)
    return onnxInferData

def resize_norm_img(img, image_shape=[3, 48, 168]):
    """车牌字符识别前：车牌图像resize、归一化、通道顺序变换，适用于OCR模型（无torch依赖）"""
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if int(np.ceil(imgH * ratio)) > imgW:
        resized_w = imgW
    else:
        resized_w = int(np.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    onnxInferData = padding_im[np.newaxis, :, :, :]
    onnxInferData = np.array(onnxInferData, dtype=np.float32)
    return onnxInferData