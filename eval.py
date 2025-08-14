from infer_onnx import RTDETROnnx, DatasetEvaluator
from utils.logging_config import setup_logger

setup_logger()

# 使用新的统一API
# 方法1: 直接使用RTDETROnnx类
detector = RTDETROnnx(onnx_path='/home/tyjt/桌面/rtdetr-l.onnx')

# 方法2: 使用工厂函数 (推荐)
# detector = create_detector('rtdetr', '/home/tyjt/桌面/rtdetr-l.onnx')

# 使用统一的评估器
evaluator = DatasetEvaluator(detector)
evaluator.evaluate_dataset(
    dataset_path='../yolo_dataset',
    conf_threshold=0.25  # 与Ultralytics对齐，避免0.001被重置为0.25导致的结果不一致
)