import argparse
from infer_onnx import DatasetEvaluator, create_detector
from utils.logging_config import setup_logger

def main():
    parser = argparse.ArgumentParser(description='评估ONNX模型在数据集上的性能')
    parser.add_argument('--model-type', type=str, default='rtdetr', 
                        choices=['rtdetr', 'yolo', 'rfdetr'], help='模型类型')
    parser.add_argument('--model-path', type=str, required=True,
                        help='ONNX模型文件路径')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.7,
                        help='IoU阈值')
    parser.add_argument('--max-images', type=int, default=None,
                        help='最大处理图片数量')
    parser.add_argument('--exclude-files', type=str, nargs='*', default=None,
                        help='需要排除的文件列表')
    parser.add_argument('--exclude-labels-containing', type=str, nargs='*', default=None,
                        help='需要排除的标签内容关键词列表')
    
    args = parser.parse_args()
    
    setup_logger()
    
    # 使用工厂函数创建检测器
    detector = create_detector(args.model_type, args.model_path)
    
    # 使用统一的评估器
    evaluator = DatasetEvaluator(detector)
    
    # 构建evaluate_dataset的参数字典
    eval_kwargs = {
        'dataset_path': args.dataset_path,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold
    }
    
    # 添加可选参数
    if args.max_images is not None:
        eval_kwargs['max_images'] = args.max_images
    if args.exclude_files is not None:
        eval_kwargs['exclude_files'] = args.exclude_files
    if args.exclude_labels_containing is not None:
        eval_kwargs['exclude_labels_containing'] = args.exclude_labels_containing
    
    evaluator.evaluate_dataset(**eval_kwargs)

if __name__ == '__main__':
    main()