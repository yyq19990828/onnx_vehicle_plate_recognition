# 目标检测模型评估指南

本系统提供了完整的目标检测模型评估功能，支持YOLO格式数据集的标准指标计算。

## 系统架构

```
onnx_vehicle_plate_recognition/
├── utils/
│   ├── detection_metrics.py      # 核心指标计算模块
│   └── output_transforms.py      # 输出格式转换函数
├── infer_onnx/
│   └── yolo_models.py            # 统一的YOLO模型API
├── demo_evaluation.py            # 评估演示脚本
└── test_metrics.py              # 指标计算测试
```

## 主要特性

1. **标准检测指标**: mAP@0.5、mAP@0.5:0.95、精度、召回率、F1分数
2. **多模型支持**: 原生支持YOLO和RF-DETR，可通过转换函数支持任意模型
3. **YOLO数据集格式**: 完全兼容YOLO训练数据集格式
4. **输出格式转换**: 内置转换函数，支持不同模型的输出格式

## 快速开始

### 1. 基本评估（YOLO模型）

```python
from infer_onnx import YoloOnnx, DatasetEvaluator

# 初始化检测器
detector = YoloOnnx("models/yolo_model.onnx", conf_thres=0.25)

# 使用统一评估器
evaluator = DatasetEvaluator(detector)

# 运行评估
results = evaluator.evaluate_dataset(
    dataset_path="path/to/dataset",  # 包含images/和labels/文件夹
    conf_threshold=0.001,            # 评估时的置信度阈值
    max_images=1000                  # 限制评估图像数量（可选）
)

print(f"mAP@0.5: {results['map50']:.3f}")
print(f"mAP@0.5:0.95: {results['map']:.3f}")
```

### 2. 使用输出转换（非YOLO模型）

```python
from infer_onnx import RFDETROnnx, DatasetEvaluator
from utils.output_transforms import get_transform_function

# 初始化RF-DETR检测器
detector = RFDETROnnx("models/rf-detr.onnx")
evaluator = DatasetEvaluator(detector)

# 获取预定义的转换函数
transform_fn = get_transform_function('rfdetr')

# 运行评估
results = evaluator.evaluate_dataset(
    dataset_path="path/to/dataset",
    output_transform=transform_fn,  # 指定输出转换函数
    conf_threshold=0.001
)
```

### 3. 自定义转换函数

```python
def my_custom_transform(detections, original_shape):
    """
    将自定义模型输出转换为YOLO格式
    输入: 任意格式的检测结果
    输出: [x1, y1, x2, y2, confidence, class_id] 格式
    """
    # 实现你的转换逻辑
    converted_detections = []
    for detection_batch in detections:
        # 转换每个批次的检测结果
        # ...转换逻辑...
        converted_detections.append(converted_batch)
    
    return converted_detections

# 使用自定义转换函数
results = detector.evaluate_dataset(
    dataset_path="path/to/dataset",
    output_transform=my_custom_transform
)
```

## 数据集格式

### 目录结构（按优先级）

系统会按以下优先级自动检测YOLO数据集结构：

**1. Test数据集（最高优先级）**
```
dataset/
├── images/
│   └── test/            # 测试图像文件
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── test/            # 测试标签文件
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**2. Validation数据集（中等优先级）**
```
dataset/
├── images/
│   └── val/             # 验证图像文件
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── val/             # 验证标签文件
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**3. Train数据集（较低优先级）**
```
dataset/
├── images/
│   └── train/           # 训练图像文件
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── train/           # 训练标签文件
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**4. 根目录结构（最低优先级）**
```
dataset/
├── images/              # 图像文件（根目录）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/              # YOLO格式标签文件（根目录）
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**注意**: 系统会自动按优先级检测，优先使用test数据集进行评估，这符合标准的模型评估实践。

### 标签格式
每个标签文件包含多行，每行格式为：
```
class_id center_x center_y width height
```
- 所有坐标值都是归一化的（0-1之间）
- `center_x, center_y`: 边界框中心点的归一化坐标
- `width, height`: 边界框的归一化宽度和高度

### 数据配置文件（可选）
```yaml
# data.yaml
names:
  0: person
  1: bicycle
  2: car
  # ... 更多类别
```

## 评估指标说明

### 核心指标

- **mAP@0.5**: IoU阈值为0.5时的平均精度均值
- **mAP@0.5:0.95**: IoU阈值从0.5到0.95（步长0.05）的平均精度均值
- **Precision (P)**: 精确度，预测为正例中实际为正例的比例
- **Recall (R)**: 召回率，实际正例中被正确预测的比例
- **F1-Score**: 精确度和召回率的调和平均数

### 计算过程

1. **IoU计算**: 计算预测框与真实框之间的交并比
2. **匹配策略**: 在不同IoU阈值下匹配预测和真实标签
3. **TP/FP统计**: 统计真正例和假正例
4. **AP计算**: 基于Precision-Recall曲线计算平均精度
5. **mAP汇总**: 对所有类别求平均得到mAP

## 内置转换函数

系统提供了以下预定义的转换函数：

| 函数名 | 用途 | 输入格式 | 输出格式 |
|--------|------|----------|----------|
| `rfdetr` | RF-DETR模型 | 标准YOLO格式 | 标准YOLO格式 |
| `yolov5` | YOLOv5模型 | 标准YOLO格式 | 标准YOLO格式 |
| `normalize_bbox` | 归一化坐标转换 | 归一化xyxy | 绝对坐标xyxy |
| `cxcywh_to_xyxy` | 中心点格式转换 | [cx,cy,w,h,...] | [x1,y1,x2,y2,...] |

使用方法：
```python
from utils.output_transforms import get_transform_function

transform_fn = get_transform_function('cxcywh_to_xyxy')
```

## 性能优化建议

1. **批量处理**: 系统支持批量处理多张图像
2. **置信度过滤**: 合理设置`conf_threshold`以平衡精度和召回率
3. **图像数量限制**: 使用`max_images`参数控制评估范围
4. **内存管理**: 大数据集评估时注意内存使用情况

## 故障排除

### 常见问题

1. **数据集路径错误**
   ```
   ValueError: 图像目录不存在: /path/to/images
   ```
   解决：检查数据集路径是否正确，确保包含images/和labels/文件夹

2. **标签文件缺失**
   ```
   Warning: 无法找到标签文件
   ```
   解决：确保每个图像都有对应的标签文件，文件名相同（不含扩展名）

3. **模型输出格式不匹配**
   ```
   Error: 检测结果格式错误
   ```
   解决：检查输出转换函数是否正确，确保输出为[x1,y1,x2,y2,conf,class_id]格式

4. **内存不足**
   ```
   MemoryError: 内存不足
   ```
   解决：使用`max_images`参数限制评估图像数量，或减小batch size

### 调试技巧

1. **检查检测结果**：
   ```python
   detections, _ = detector(image)
   print(f"检测结果形状: {detections[0].shape}")
   print(f"前5个检测: {detections[0][:5]}")
   ```

2. **验证转换函数**：
   ```python
   transformed = transform_function(detections, original_shape)
   assert transformed[0].shape[1] == 6  # 确保6列输出
   ```

3. **小规模测试**：
   ```python
   results = detector.evaluate_dataset(dataset_path, max_images=10)
   ```

## 扩展开发

### 添加新的转换函数

1. 在`utils/output_transforms.py`中实现新函数
2. 添加到`TRANSFORM_FUNCTIONS`字典
3. 编写文档和测试

### 添加新的评估指标

1. 在`utils/detection_metrics.py`中实现新指标
2. 更新`DetectionMetrics.process()`方法
3. 更新`print_metrics()`函数

## 完整示例

参考`demo_evaluation.py`文件中的完整示例代码，包含：
- YOLO模型评估
- RF-DETR模型评估  
- 自定义转换函数使用
- 错误处理和日志记录