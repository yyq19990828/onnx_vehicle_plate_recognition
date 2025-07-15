import cv2
import numpy as np
import os
import logging
from PIL import Image, ImageDraw, ImageFont

def draw_detections(image, detections, class_names, colors, plate_results=None, font_path="SourceHanSans-VF.ttf"):
    """
    Draws detection boxes on the image. Handles Chinese characters gracefully.
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    img_h, img_w = pil_img.height, pil_img.width
    
    font_found = os.path.exists(font_path)
    if font_found:
        try:
            # 尝试加载粗体字重
            # Removed layout_engine for compatibility with older Pillow versions.
            font = ImageFont.truetype(font_path, 20)
            font.set_variation_by_name('Bold')
        except (IOError, ValueError, AttributeError):
            try:
                # 如果失败，回退到常规字体
                font = ImageFont.truetype(font_path, 20)
                logging.warning(f"Could not load bold variation for {font_path}. Using regular weight.")
            except IOError:
                logging.warning(f"Could not load font file at {font_path}. Using default font.")
                font = ImageFont.load_default()
                font_found = False # Treat as not found if loading fails
    else:
        logging.warning(f"Font file not found at {font_path}. Chinese characters will not be displayed. Using default font.")
        font = ImageFont.load_default()

    for idx, det in enumerate(detections):
        for j, (*xyxy, conf, cls) in enumerate(det):
            class_id = int(cls)
            label = f'{class_names[class_id] if class_id < len(class_names) else "unknown"} {conf:.2f}'
            x1, y1, x2, y2 = map(int, xyxy)
            
            # 根据类别ID选择颜色
            box_color = colors[class_id % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3) # 加粗边框
            
            # 绘制类别标签
            label_y_pos = y1 - 25
            if label_y_pos < 0: label_y_pos = y1 + 5 # 如果标签超出顶部，则移到框内
            draw.text((x1, label_y_pos), label, fill=box_color, font=font)
            
            # 绘制OCR结果 - 只对车牌类别的检测框绘制OCR信息
            if (class_names[class_id] == 'plate' and plate_results and 
                j < len(plate_results) and plate_results[j] is not None):
                plate_info = plate_results[j]
                ocr_str = plate_info.get("plate_text", "")
                color_str = plate_info.get("color", "")
                layer_str = plate_info.get("layer", "")
                
                if ocr_str and font_found:
                    # 构建OCR文本，只显示有意义的信息
                    ocr_parts = []
                    if ocr_str: ocr_parts.append(ocr_str)
                    if color_str and color_str != "unknown": ocr_parts.append(color_str)
                    if layer_str and layer_str != "unknown": ocr_parts.append(layer_str)
                    ocr_text = ' '.join(ocr_parts)
                    
                    if ocr_text:  # 只有当有有效文本时才绘制
                        try:
                            # 使用 getbbox 替换 getsize
                            text_bbox = draw.textbbox((0, 0), ocr_text, font=font)
                            text_w = text_bbox[2] - text_bbox[0]
                            text_h = text_bbox[3] - text_bbox[1]
                        except AttributeError:
                            # 兼容旧版 Pillow
                            text_w, text_h = draw.textsize(ocr_text, font=font)

                        # 改进的文本放置逻辑
                        # 垂直位置：优先放在框的下方，如果空间不够则放在上方
                        text_y = y2 + 8  # 在框下方，留8px间距
                        if text_y + text_h > img_h:  # 如果超出底部边界
                            text_y = y1 - text_h - 8  # 移到框上方，留8px间距
                            if text_y < 0:  # 如果上方也没有空间
                                text_y = max(0, y1 + 5)  # 放在框内顶部，但不超出图像边界

                        # 水平位置：居中对齐到检测框
                        box_center_x = (x1 + x2) // 2
                        text_x = box_center_x - text_w // 2  # 文本居中
                        
                        # 确保文本不超出图像边界
                        if text_x < 0:
                            text_x = 0
                        elif text_x + text_w > img_w:
                            text_x = img_w - text_w
                        
                        # 绘制文本背景以提高可读性
                        padding = 2
                        bg_x1 = max(0, text_x - padding)
                        bg_y1 = max(0, text_y - padding)
                        bg_x2 = min(img_w, text_x + text_w + padding)
                        bg_y2 = min(img_h, text_y + text_h + padding)
                        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="white", outline="black")
                        
                        # 绘制OCR文本
                        draw.text((text_x, text_y), ocr_text, fill="blue", font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)