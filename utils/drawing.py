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
                
                # 检查是否应该显示OCR信息
                if not plate_info.get("should_display_ocr", False) or not font_found:
                    continue

                ocr_str = plate_info.get("plate_text", "")
                color_str = plate_info.get("color", "")
                layer_str = plate_info.get("layer", "")

                # 构建第二行的信息文本
                info_parts = []
                if color_str and color_str != "unknown": info_parts.append(color_str)
                if layer_str and layer_str != "unknown": info_parts.append(layer_str)
                info_str = ' '.join(info_parts)

                # 只有在车牌号存在时才绘制
                if ocr_str:
                    try:
                        # 使用 textbbox 获取更准确的尺寸
                        line1_bbox = draw.textbbox((0, 0), ocr_str, font=font)
                        line1_w, line1_h = line1_bbox[2] - line1_bbox[0], line1_bbox[3] - line1_bbox[1]
                        
                        line2_w, line2_h = 0, 0
                        if info_str:
                            line2_bbox = draw.textbbox((0, 0), info_str, font=font)
                            line2_w, line2_h = line2_bbox[2] - line2_bbox[0], line2_bbox[3] - line2_bbox[1]

                    except AttributeError:
                        # 兼容旧版 Pillow
                        line1_w, line1_h = draw.textsize(ocr_str, font=font)
                        line2_w, line2_h = (draw.textsize(info_str, font=font) if info_str else (0, 0))

                    total_text_h = line1_h + (line2_h + 5 if info_str else 0) # 两行文本的总高度，带间距
                    max_text_w = max(line1_w, line2_w)

                    # 垂直位置：优先放在框下方
                    text_y = y2 + 8 # 框下方8px
                    if text_y + total_text_h > img_h: # 如果下方空间不足，移到上方
                        text_y = y1 - total_text_h - 8
                        if text_y < 0: # 如果上方也超出
                            text_y = max(0, y1 + 5) # 放在框内顶部

                    # 水平位置：居中对齐
                    box_center_x = (x1 + x2) // 2
                    text_x1 = box_center_x - line1_w // 2
                    text_x2 = box_center_x - line2_w // 2 if info_str else 0
                    
                    # 确保背景框不超出图像边界
                    padding = 3
                    bg_x1 = max(0, box_center_x - max_text_w // 2 - padding)
                    bg_y1 = max(0, text_y - padding)
                    bg_x2 = min(img_w, box_center_x + max_text_w // 2 + padding)
                    bg_y2 = min(img_h, text_y + total_text_h + padding)
                    
                    # 绘制文本背景
                    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 180))

                    # 绘制第一行：车牌号
                    draw.text((text_x1, text_y), ocr_str, fill="blue", font=font)
                    
                    # 绘制第二行：颜色和层信息
                    if info_str:
                        draw.text((text_x2, text_y + line1_h + 5), info_str, fill="black", font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)