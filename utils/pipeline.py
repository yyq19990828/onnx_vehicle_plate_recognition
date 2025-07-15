import cv2
import numpy as np
import yaml

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
        print(f"Error initializing detector: {e}")
        print("Please ensure the ONNX model path is correct and onnxruntime is installed.")
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

    # 2. Process detections
    if detections and len(detections[0]) > 0:
        # Use a specific confidence threshold for plates if provided
        plate_conf_thres = args.plate_conf_thres if args.plate_conf_thres is not None else args.conf_thres
        
        # ROI filtering for plates
        h_img, _, _ = frame.shape
        roi_top_pixel = int(h_img * args.roi_top_ratio)

        filtered_detections = []
        for det in detections[0]:
            *xyxy, conf, cls = det
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else "unknown"
            
            # If it's a plate, check if it's within the ROI
            if class_name == 'plate':
                y1 = xyxy[1]
                if y1 < roi_top_pixel:
                    continue # Skip plate if it's above the ROI line
            
            # Apply specific confidence threshold for plates
            if class_name == 'plate' and conf < plate_conf_thres:
                continue

            filtered_detections.append(det)
        
        # Replace original detections with filtered ones
        detections[0] = np.array(filtered_detections) if filtered_detections else []


        for detection_idx, (*xyxy, conf, cls) in enumerate(detections[0]):
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else "unknown"
            
            # Keep float values for JSON output, ensuring they are standard Python floats
            float_xyxy = [float(c) for c in xyxy]
            
            # Use integer values for image processing (cropping, drawing)
            x1, y1, x2, y2 = map(int, float_xyxy)

            plate_text, plate_conf, color_str, layer_str = "", 0.0, "", ""
            plate_info = None
            
            if class_name == 'plate':
                # --- Start of modification: Enlarge bbox for OCR ---
                h_img, w_img, _ = frame.shape
                
                # Calculate expanded bbox for better OCR. The original bbox (x1,y1,x2,y2) is preserved for drawing.
                w, h = x2 - x1, y2 - y1
                exp_x1 = int(max(0, x1 - w * 0.1))
                exp_y1 = int(max(0, y1 - h * 0.1))
                exp_x2 = int(min(w_img, x2 + w * 0.1))
                exp_y2 = int(min(h_img, y2 + h * 0.1))
                
                # Crop with expanded bbox for processing
                plate_img = frame[exp_y1:exp_y2, exp_x1:exp_x2]
                
                if plate_img.size == 0:
                    continue
                # --- End of modification ---

                # Color/Layer recognition
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
                
                # OCR recognition
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
            
            plate_results.append(plate_info)

            if class_name == 'plate':
                output_data.append({
                    "plate_box2d": float_xyxy,
                    "plate_name": plate_text,
                    "plate_color": color_str,
                    "plate_layer": layer_str
                })
            else:
                # Note: Vehicle color is not detected by the current model.
                # This is a placeholder.
                vehicle_color = "unknown"
                output_data.append({
                    "type": class_name,
                    "box2d": float_xyxy,
                    "color": vehicle_color
                })

    # 3. Draw results
    if detections and len(detections[0]) > 0:
        h, w = original_shape
        detections[0][:, 0] = np.clip(detections[0][:, 0], 0, w)
        detections[0][:, 1] = np.clip(detections[0][:, 1], 0, h)
        detections[0][:, 2] = np.clip(detections[0][:, 2], 0, w)
        detections[0][:, 3] = np.clip(detections[0][:, 3], 0, h)

    result_frame = draw_detections(frame.copy(), detections, class_names, colors, plate_results=plate_results)

    return result_frame, output_data