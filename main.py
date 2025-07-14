import cv2
import numpy as np
import json
import os
import argparse

from utils.pipeline import initialize_models, process_frame

def infer_source_type(input_path):
    """
    Infer the source type based on the input path.
    """
    input_path_lower = input_path.lower()
    if os.path.isdir(input_path):
        return 'folder'
    elif any(input_path_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        return 'image'
    elif any(input_path_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
        return 'video'
    elif input_path_lower.startswith('rtsp://'):
        return 'rtsp'
    elif input_path.isdigit():
        return 'camera'
    else:
        return 'unknown'

def main(args):
    # Check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize models
    models = initialize_models(args)
    if models is None:
        return
    detector, color_layer_classifier, ocr_model, character, class_names, colors = models

    source_type = infer_source_type(args.input)

    if source_type == 'image':
        # Load image
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Could not read image {args.input}")
            return

        # Process the single image
        result_img, output_data = process_frame(
            img, detector, color_layer_classifier, ocr_model, character, class_names, colors, args
        )

        if args.output_mode == 'save':
            # Save the annotated image
            output_image_path = os.path.join(args.output_dir, os.path.basename(args.input))
            cv2.imwrite(output_image_path, result_img)
            print(f"Result image saved to {output_image_path}")

            # Save JSON results
            output_json_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input))[0] + ".json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            print(f"JSON results saved to {output_json_path}")
        elif args.output_mode == 'show':
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif source_type == 'folder':
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        for image_file in image_files:
            image_path = os.path.join(args.input, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue

            result_img, output_data = process_frame(
                img, detector, color_layer_classifier, ocr_model, character, class_names, colors, args
            )

            if args.output_mode == 'save':
                output_image_path = os.path.join(args.output_dir, image_file)
                cv2.imwrite(output_image_path, result_img)
                print(f"Result image saved to {output_image_path}")

                output_json_path = os.path.join(args.output_dir, os.path.splitext(image_file)[0] + ".json")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                print(f"JSON results saved to {output_json_path}")
            elif args.output_mode == 'show':
                cv2.imshow(f"Result - {image_file}", result_img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break # Allow quitting with 'q'
        cv2.destroyAllWindows()


    elif source_type in ['video', 'rtsp', 'camera']:
        # Setup video capture
        if source_type == 'camera':
            cap = cv2.VideoCapture(int(args.input))
        else: # video or rtsp
            cap = cv2.VideoCapture(args.input)

        if not cap.isOpened():
            print(f"Error: Could not open video source {args.input}")
            return

        # Setup video writer if saving
        writer = None
        if args.output_mode == 'save':
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = os.path.join(args.output_dir, "result.mp4")

            # Try to use a more efficient codec (H.264), with a fallback to mp4v
            fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_video_path, fourcc_h264, fps, (width, height))
            
            if not writer.isOpened():
                print("Warning: H.264 codec ('avc1') not available. Falling back to 'mp4v'. Output file may be large.")
                fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_video_path, fourcc_mp4v, fps, (width, height))

            if writer.isOpened():
                print(f"Saving result video to {output_video_path}")
            else:
                print("Error: Could not open video writer. Cannot save video.")
                # Set writer to None to avoid crashing in the loop
                writer = None

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (args.frame_skip + 1) == 0:
                # Process frame
                result_frame, _ = process_frame(
                    frame, detector, color_layer_classifier, ocr_model, character, class_names, colors, args
                )
            else:
                result_frame = frame # Use original frame if skipped

            # Output
            if args.output_mode == 'save':
                writer.write(result_frame)
            elif args.output_mode == 'show':
                cv2.imshow("Result", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Vehicle and Plate Recognition')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the ONNX detection model.')
    parser.add_argument('--input', type=str, default='data/sample.jpg', help='Path to input image/video or camera ID.')
    parser.add_argument('--output-mode', type=str, choices=['save', 'show'], default='save', help='Output mode: save to file or show in a window.')
    parser.add_argument('--frame-skip', type=int, default=0, help='Number of frames to skip between processing.')
    parser.add_argument('--output-dir', type=str, default='runs', help='Directory to save output results.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold for detection.')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IoU threshold for NMS.')
    parser.add_argument('--plate-conf-thres', type=float, default=None, help='Specific confidence threshold for plates.')
    parser.add_argument('--color-layer-model', type=str, default='models/color_layer.onnx', help='Path to color/layer ONNX model.')
    parser.add_argument('--ocr-model', type=str, default='models/ocr.onnx', help='Path to OCR ONNX model.')
    parser.add_argument('--ocr-dict-yaml', type=str, default='models/ocr_dict.yaml', help='Path to OCR dict YAML file.')
    
    args = parser.parse_args()
    
    # Create a dummy model file if it doesn't exist, as we don't have a real one yet.
    if not os.path.exists(args.model_path):
        print(f"Warning: Model file not found at {args.model_path}. A real model is needed for inference.")
        # In a real scenario, you would not create a dummy file. This is for testing the script structure.
        # To run this script, you must provide a valid ONNX model.
    
    # Create a dummy sample image if it doesn't exist and input is an image
    source_type = infer_source_type(args.input)
    if source_type == 'image' and not os.path.exists(args.input):
        if not os.path.exists('data'):
            os.makedirs('data')
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(args.input, dummy_image)
        print(f"Created a dummy sample image at {args.input}")

    main(args)