#!/bin/bash

python main.py \
    --model-path models/2024080100.onnx \
    --input data/R1_Ds_CamE-20250626092740-549.mp4 \
    --output-mode show \
    --source-type video

# python main.py --model-path models/2024080100.onnx