import cv2
import numpy as np
import json
import os
import argparse
import yaml
from PIL import Image, ImageDraw, ImageFont

from infer_onnx.det_onnx import DetONNX
from infer_onnx.ocr_onnx import ColorLayerONNX, OCRONNX
from utils.ocr_image_processing import process_plate_image, image_pretreatment, resize_norm_img
from utils.ocr_post_processing import decode
from utils.drawing import draw_detections

def main(args):
    # Check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load image
    img = cv2.imread(args.input_image)
    if img is None:
        print(f"Error: Could not read image {args.input_image}")
        return

    # Initialize the detector
    try:
        detector = DetONNX(args.model_path)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        print("Please ensure the ONNX model path is correct and onnxruntime is installed.")
        return

    # 初始化颜色层数与OCR模型
    color_layer_model_path = getattr(args, "color_layer_model", "models/color_layer.onnx")
    ocr_model_path = getattr(args, "ocr_model", "models/ocr.onnx")
    ocr_dict_yaml_path = getattr(args, "ocr_dict_yaml", "models/ocr_dict.yaml")
    with open(ocr_dict_yaml_path, "r", encoding="utf-8") as f:
        dict_yaml = yaml.safe_load(f)
        character = ["blank"] + dict_yaml["ocr_dict"] + [" "]
    color_layer_classifier = ColorLayerONNX(color_layer_model_path)
    ocr_model = OCRONNX(ocr_model_path)

    # 1. Object Detection
    print("Running detection...")
    detections, original_shape = detector(img)
    print(f"Found {len(detections[0]) if detections else 0} objects.")

    # 从配置文件读取类别
    with open("models/det_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    class_names = config["class_names"]
    colors = config["visual_colors"]
    
    output_data = {
        "detections": []
    }
    plate_results = []  # 存储每个检测框对应的车牌信息，包括非车牌检测

    # 2. Process detections
    if detections and len(detections[0]) > 0:
        for detection_idx, (*xyxy, conf, cls) in enumerate(detections[0]):
            x1, y1, x2, y2 = map(int, xyxy)
            # 裁剪车牌区域
            plate_img = img[y1:y2, x1:x2]

            # Add a check to prevent crashing on empty crops from invalid bboxes
            if plate_img.size == 0:
                continue

            class_name = class_names[int(cls)] if int(cls) < len(class_names) else "unknown"
            
            plate_text, plate_conf, color_str, layer_str = "", 0.0, "", ""
            plate_info = None  # 初始化车牌信息
            
            if class_name == 'plate':
                # 颜色/层数识别
                img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                color_input = image_pretreatment(img_rgb)
                preds_color, preds_layer = color_layer_classifier.infer(color_input)
                color_index = int(np.argmax(preds_color))
                layer_index = int(np.argmax(preds_layer))
                # 从yaml加载颜色和层数字典
                with open("models/plate_color_layer.yaml", "r", encoding="utf-8") as f:
                    color_layer_yaml = yaml.safe_load(f)
                color_dict = color_layer_yaml["color_dict"]
                layer_dict = color_layer_yaml["layer_dict"]
                color_str = color_dict.get(color_index, "unknown")
                layer_str = layer_dict.get(layer_index, "unknown")
                # 双层处理
                is_double = (layer_str == "double")
                processed_plate = process_plate_image(plate_img, is_double_layer=is_double)
                # OCR识别
                ocr_input = resize_norm_img(processed_plate)
                ocr_out = ocr_model.infer(ocr_input)
                preds_idx = np.asarray(ocr_out[0]).argmax(axis=2)
                preds_prob = np.asarray(ocr_out[0]).max(axis=2)
                ocr_result = decode(character, preds_idx, preds_prob, is_remove_duplicate=True)
                plate_text = ocr_result[0][0] if ocr_result else ""
                plate_conf = float(ocr_result[0][1]) if ocr_result else 0.0
                
                plate_info = {
                    "plate_text": plate_text,
                    "color": color_str,
                    "layer": layer_str
                }
            
            # 为每个检测结果存储对应的车牌信息（如果不是车牌则为None）
            plate_results.append(plate_info)

            # 结果保存
            output_data["detections"].append({
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls),
                "class_name": class_name,
                "plate_text": plate_text,
                "plate_conf": plate_conf,
                "color": color_str,
                "layer": layer_str
            })

    # 3. Draw results
    # Clip boxes to image boundaries
    if detections and len(detections[0]) > 0:
        h, w = original_shape
        detections[0][:, 0] = np.clip(detections[0][:, 0], 0, w)
        detections[0][:, 1] = np.clip(detections[0][:, 1], 0, h)
        detections[0][:, 2] = np.clip(detections[0][:, 2], 0, w)
        detections[0][:, 3] = np.clip(detections[0][:, 3], 0, h)

    result_img = draw_detections(img.copy(), detections, class_names, colors, plate_results=plate_results)

    # Save the annotated image
    output_image_path = os.path.join(args.output_dir, "result.jpg")
    cv2.imwrite(output_image_path, result_img)
    print(f"Result image saved to {output_image_path}")

    # Save JSON results
    output_json_path = os.path.join(args.output_dir, "result.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"JSON results saved to {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Vehicle and Plate Recognition')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the ONNX detection model.')
    parser.add_argument('--input-image', type=str, default='data/sample.jpg', help='Path to the input image.')
    parser.add_argument('--output-dir', type=str, default='runs', help='Directory to save output results.')
    parser.add_argument('--color-layer-model', type=str, default='models/color_layer.onnx', help='Path to color/layer ONNX model.')
    parser.add_argument('--ocr-model', type=str, default='models/ocr.onnx', help='Path to OCR ONNX model.')
    parser.add_argument('--ocr-dict-yaml', type=str, default='models/ocr_dict.yaml', help='Path to OCR dict YAML file.')
    
    args = parser.parse_args()
    
    # Create a dummy model file if it doesn't exist, as we don't have a real one yet.
    if not os.path.exists(args.model_path):
        print(f"Warning: Model file not found at {args.model_path}. A real model is needed for inference.")
        # In a real scenario, you would not create a dummy file. This is for testing the script structure.
        # To run this script, you must provide a valid ONNX model.
    
    # Create a dummy sample image if it doesn't exist
    if not os.path.exists(args.input_image):
        if not os.path.exists('data'):
            os.makedirs('data')
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(args.input_image, dummy_image)
        print(f"Created a dummy sample image at {args.input_image}")

    main(args)