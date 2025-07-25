import cv2
import numpy as np
import yaml
import logging

from utils import (
    process_plate_image,
    image_pretreatment,
    resize_norm_img,
    decode,
    draw_detections
)

def initialize_models(args):
    """
    Initialize all the models required for the pipeline.
    """
    # Initialize the detector
    try:
        from infer_onnx import DetONNX
        detector = DetONNX(
            args.model_path,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
    except Exception as e:
        logging.error(f"Error initializing detector: {e}")
        logging.error("Please ensure the ONNX model path is correct and onnxruntime is installed.")
        return None, None, None, None

    # Initialize color/layer and OCR models
    from infer_onnx import ColorLayerONNX, OCRONNX
    color_layer_model_path = getattr(args, "color_layer_model", "models/color_layer.onnx")
    ocr_model_path = getattr(args, "ocr_model", "models/ocr.onnx")
    ocr_dict_yaml_path = getattr(args, "ocr_dict_yaml", "models/ocr_dict.yaml")
    
    with open(ocr_dict_yaml_path, "r", encoding="utf-8") as f:
        dict_yaml = yaml.safe_load(f)
        character = ["blank"] + dict_yaml["ocr_dict"] + [" "]
        
    color_layer_classifier = ColorLayerONNX(color_layer_model_path)
    ocr_model = OCRONNX(ocr_model_path)

    # Load class names and colors from config
    with open("models/det_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    class_names = config["class_names"]
    colors = config["visual_colors"]

    return detector, color_layer_classifier, ocr_model, character, class_names, colors

def process_frame(frame, detector, color_layer_classifier, ocr_model, character, class_names, colors, args):
    """
    Process a single frame for vehicle and plate detection and recognition.
    """
    # 1. Object Detection
    detections, original_shape = detector(frame)

    output_data = []
    plate_results = []

    # 2. Process all detections to gather data for JSON and prepare for drawing
    if detections and len(detections[0]) > 0:
        h_img, w_img, _ = frame.shape
        roi_top_pixel = int(h_img * args.roi_top_ratio)

        # Ensure detections are clipped within frame boundaries
        clipped_detections = detections[0].copy()
        clipped_detections[:, 0] = np.clip(clipped_detections[:, 0], 0, w_img)
        clipped_detections[:, 1] = np.clip(clipped_detections[:, 1], 0, h_img)
        clipped_detections[:, 2] = np.clip(clipped_detections[:, 2], 0, w_img)
        clipped_detections[:, 3] = np.clip(clipped_detections[:, 3], 0, h_img)
        
        plate_conf_thres = args.plate_conf_thres if args.plate_conf_thres is not None else args.conf_thres

        for detection_idx, (*xyxy, conf, cls) in enumerate(clipped_detections):
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else "unknown"

            # Apply specific confidence threshold for plates
            if class_name == 'plate' and conf < plate_conf_thres:
                plate_results.append(None) # Keep lists in sync
                continue

            # Keep float values for JSON output
            float_xyxy = [float(c) for c in xyxy]
            x1, y1, x2, y2 = map(int, float_xyxy)
            w, h = x2 - x1, y2 - y1

            plate_text, color_str, layer_str = "", "", ""
            plate_info = None

            if class_name == 'plate':
                exp_x1 = int(max(0, x1 - w * 0.1))
                exp_y1 = int(max(0, y1 - h * 0.1))
                exp_x2 = int(min(w_img, x2 + w * 0.1))
                exp_y2 = int(min(h_img, y2 + h * 0.1))
                plate_img = frame[exp_y1:exp_y2, exp_x1:exp_x2]

                if plate_img.size > 0:
                    img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    color_input = image_pretreatment(img_rgb)
                    preds_color, preds_layer = color_layer_classifier.infer(color_input)
                    color_index = int(np.argmax(preds_color))
                    layer_index = int(np.argmax(preds_layer))

                    with open("models/plate_color_layer.yaml", "r", encoding="utf-8") as f:
                        color_layer_yaml = yaml.safe_load(f)
                    color_dict = color_layer_yaml["color_dict"]
                    layer_dict = color_layer_yaml["layer_dict"]
                    color_str = color_dict.get(color_index, "unknown")
                    layer_str = layer_dict.get(layer_index, "unknown")

                    is_double = (layer_str == "double")
                    processed_plate = process_plate_image(plate_img, is_double_layer=is_double)
                    ocr_input = resize_norm_img(processed_plate)
                    ocr_out = ocr_model.infer(ocr_input)
                    preds_idx = np.asarray(ocr_out[0]).argmax(axis=2)
                    preds_prob = np.asarray(ocr_out[0]).max(axis=2)
                    ocr_result = decode(character, preds_idx, preds_prob, is_remove_duplicate=True)
                    plate_text = ocr_result[0][0] if ocr_result else ""
                    
                    # Determine if OCR text should be displayed based on ROI and width
                    should_display_ocr = (y1 >= roi_top_pixel) and (w > 50)
                    
                    plate_info = {
                        "plate_text": plate_text, "color": color_str, "layer": layer_str,
                        "should_display_ocr": should_display_ocr
                    }
            
            plate_results.append(plate_info)

            # Populate JSON data regardless of display logic
            if class_name == 'plate':
                output_data.append({
                    "plate_box2d": float_xyxy, "plate_name": plate_text,
                    "plate_color": color_str, "plate_layer": layer_str,
                    "width": w, "height": h
                })
            else:
                output_data.append({
                    "type": class_name, "box2d": float_xyxy, "color": "unknown",
                    "width": w, "height": h
                })

    # 3. Draw detections
    # The original detections array is used; filtering of what to draw is handled by the drawing function.
    result_frame = draw_detections(frame.copy(), detections, class_names, colors, plate_results=plate_results)

    return result_frame, output_data