# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run the main script**: `python main.py` with various arguments. Refer to `python main.py --help` for full options.
  - Example (video input, show output): `bash run.sh` or `python main.py --model-path models/2024080100.onnx --input data/R1_Ds_CamE-20250626092740-549.mp4 --output-mode show --source-type video`
  - Example (image input, save output): `python main.py --model-path models/yolov8s_640.onnx --input data/sample.jpg --source-type image --output-mode save`
  - Example (camera input, show output): `python main.py --model-path models/yolov8s_640.onnx --input 0 --source-type camera --output-mode show`

## High-level Architecture

This project implements a vehicle and license plate recognition system using ONNX models.

-   **Input Handling**: Supports image, video files, and camera input. Handled by `main.py`.
-   **Model Inference**:
    -   `det_onnx.py`: Handles vehicle and plate detection using a general object detection model (e.g., YOLO).
    -   `ocr_onnx.py`: Handles OCR for license plates and color/layer classification.
-   **Models**: ONNX models for detection, color/layer classification, and OCR are located in the `models/` directory, along with their respective configuration YAML files (`det_config.yaml`, `ocr_dict.yaml`, `plate_color_layer.yaml`).
-   **Utilities**: The `utils/` directory contains helper functions for drawing results (`drawing.py`), image processing (`image_processing.py`, `ocr_image_processing.py`), Non-Maximum Suppression (`nms.py`), OCR post-processing (`ocr_post_processing.py`), and the main processing pipeline (`pipeline.py`).
-   **Output**: Results can be saved to the `runs/` directory (annotated image/video and a JSON file with detailed detection information) or displayed in real-time.