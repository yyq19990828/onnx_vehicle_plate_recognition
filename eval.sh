#!/usr/bin/env zsh

# 简单脚本：直接将 eval.py 的参数写入脚本内

python3 eval.py \
    --model-type rfdetr \
    --model-path models/rfdetr-20250811.onnx \
    --dataset-path /home/tyjt/桌面/yolo_dataset \
    --conf-threshold 0.25 \
    --iou-threshold 0.7