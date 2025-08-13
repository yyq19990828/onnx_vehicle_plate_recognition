from infer_onnx import RFDETROnnx, DetONNX
import logging
from utils.logging_config import setup_logger

setup_logger()

detector = RFDETROnnx(onnx_path='/home/hehao/桌面/rf-detr/best_ema.onnx')

detector.evaluate_dataset(
    dataset_path='/media/hehao/data/yiqing/dataset/车型检测/ruqi_wuxi0728_yolo',
    conf_threshold=0.001  # 设置置信度阈值为0.001，过滤低置信度检测
)

detector = DetONNX(onnx_path='/media/hehao/data/yiqing/dataset/ultralytics/2024080100.onnx')

detector.evaluate_dataset(
    dataset_path='/media/hehao/data/yiqing/dataset/车型检测/ruqi_wuxi0728_yolo',
    conf_threshold=0.001  # 设置置信度阈值为0.001，过滤低置信度检测
)