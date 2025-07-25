import cv2
import numpy as np
import json
import os
import argparse
import logging

from utils.pipeline import initialize_models, process_frame
from utils.logging_config import setup_logger

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
    # Setup logger
    setup_logger(args.log_level)

    # Check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

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
            logging.error(f"Could not read image {args.input}")
            return

        # Process the single image
        result_img, output_data = process_frame(
            img, detector, color_layer_classifier, ocr_model, character, class_names, colors, args
        )

        if args.output_mode == 'save':
            # Save the annotated image
            output_image_path = os.path.join(args.output_dir, os.path.basename(args.input))
            cv2.imwrite(output_image_path, result_img)
            logging.info(f"Result image saved to {output_image_path}")

            # Save JSON results
            output_json_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input))[0] + ".json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logging.info(f"JSON results saved to {output_json_path}")
        elif args.output_mode == 'show':
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif source_type == 'folder':
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        total_images = len(image_files)
        logging.info(f"Found {total_images} images in folder '{args.input}'.")
        for i, image_file in enumerate(image_files):
            logging.info(f"Processing image {i + 1}/{total_images}: {image_file}")
            image_path = os.path.join(args.input, image_file)
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Could not read image {image_path}, skipping.")
                continue

            result_img, output_data = process_frame(
                img, detector, color_layer_classifier, ocr_model, character, class_names, colors, args
            )

            if args.output_mode == 'save':
                output_image_path = os.path.join(args.output_dir, image_file)
                cv2.imwrite(output_image_path, result_img)
                # Log saving action, but the progress is already logged above
                # logging.info(f"Result image saved to {output_image_path}")

                output_json_path = os.path.join(args.output_dir, os.path.splitext(image_file)[0] + ".json")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                # logging.info(f"JSON results saved to {output_json_path}")
            elif args.output_mode == 'show':
                cv2.imshow(f"Result - {image_file}", result_img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break # Allow quitting with 'q'
        cv2.destroyAllWindows()
        logging.info(f"Finished processing all {total_images} images.")


    elif source_type in ['video', 'rtsp', 'camera']:
        # Setup video capture
        if source_type == 'camera':
            cap = cv2.VideoCapture(int(args.input))
        else: # video or rtsp
            cap = cv2.VideoCapture(args.input)

        if not cap.isOpened():
            logging.error(f"Could not open video source {args.input}")
            return

        # Setup video writer if saving
        writer = None
        if args.output_mode == 'save':
            # Create a directory for the video's output
            video_name = os.path.splitext(os.path.basename(args.input))[0] if source_type == 'video' else 'camera_output'
            video_output_dir = os.path.join(args.output_dir, video_name)
            frames_dir = os.path.join(video_output_dir, 'frames')
            json_dir = os.path.join(video_output_dir, 'json')

            os.makedirs(video_output_dir, exist_ok=True)
            if args.save_frame:
                os.makedirs(frames_dir, exist_ok=True)
            if args.save_json:
                os.makedirs(json_dir, exist_ok=True)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Use the same name as the input file for the output video
            output_filename = os.path.basename(args.input) if source_type == 'video' else 'result.mp4'
            output_video_path = os.path.join(video_output_dir, output_filename)

            # Try to use a more efficient codec (H.264), with a fallback to mp4v
            fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_video_path, fourcc_h264, fps, (width, height))
            
            if not writer.isOpened():
                logging.warning("H.264 codec ('avc1') not available. Falling back to 'mp4v'. Output file may be large.")
                fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_video_path, fourcc_mp4v, fps, (width, height))

            if writer.isOpened():
                logging.info(f"Saving result video to {output_video_path}")
            else:
                logging.error("Could not open video writer. Cannot save video.")
                # Set writer to None to avoid crashing in the loop
                writer = None

        # Get total frame count for progress logging
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total_frames = 0 # Use 0 to indicate unknown total

        if total_frames > 0:
            logging.info(f"Processing video with {total_frames} frames.")
        else:
            logging.info("Processing video stream (total frames unknown).")

        frame_count = 0
        last_result_frame = None  # 保存上一次的检测结果
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (args.frame_skip + 1) == 0:
                # Save original frame if requested
                if args.save_frame:
                    frame_filename = f"{video_name}_{frame_count:06d}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)

                # Process frame
                result_frame, output_data = process_frame(
                    frame, detector, color_layer_classifier, ocr_model, character, class_names, colors, args
                )
                last_result_frame = result_frame.copy()  # 保存检测结果

                # Save JSON data if requested
                if args.save_json:
                    json_filename = f"{video_name}_{frame_count:06d}.json"
                    json_path = os.path.join(json_dir, json_filename)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=4)
            else:
                # 如果跳帧，继续使用上一次的检测结果
                if last_result_frame is not None:
                    result_frame = last_result_frame
                else:
                    result_frame = frame  # 如果还没有检测结果，使用原始帧

            # Log progress every 100 frames
            if frame_count > 0 and frame_count % 100 == 0:
                if total_frames > 0:
                    logging.info(f"Processed frame {frame_count}/{total_frames}...")
                else:
                    logging.info(f"Processed frame {frame_count}...")

            # Output
            if args.output_mode == 'save':
                if writer:  # 添加安全检查
                    writer.write(result_frame)
            elif args.output_mode == 'show':
                cv2.imshow("Result", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        logging.info("Finished processing video.")
        
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
    parser.add_argument('--roi-top-ratio', type=float, default=0.5, help='The top ratio of the ROI for detection, range [0.0, 1.0]. Default is 0.5, meaning the lower half of the image.')
    parser.add_argument('--plate-conf-thres', type=float, default=None, help='Specific confidence threshold for plates.')
    parser.add_argument('--color-layer-model', type=str, default='models/color_layer.onnx', help='Path to color/layer ONNX model.')
    parser.add_argument('--ocr-model', type=str, default='models/ocr.onnx', help='Path to OCR ONNX model.')
    parser.add_argument('--ocr-dict-yaml', type=str, default='models/ocr_dict.yaml', help='Path to OCR dict YAML file.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.')
    parser.add_argument('--save-frame', action='store_true', help='Save processed frames as images for video input.')
    parser.add_argument('--save-json', action='store_true', help='Save JSON results for each processed frame for video input.')
    
    args = parser.parse_args()
    
    # Create a dummy model file if it doesn't exist, as we don't have a real one yet.
    if not os.path.exists(args.model_path):
        logging.warning(f"Model file not found at {args.model_path}. A real model is needed for inference.")
        # In a real scenario, you would not create a dummy file. This is for testing the script structure.
        # To run this script, you must provide a valid ONNX model.
    
    # Create a dummy sample image if it doesn't exist and input is an image
    source_type = infer_source_type(args.input)
    if source_type == 'image' and not os.path.exists(args.input):
        if not os.path.exists('data'):
            os.makedirs('data')
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(args.input, dummy_image)
        logging.info(f"Created a dummy sample image at {args.input}")

    main(args)