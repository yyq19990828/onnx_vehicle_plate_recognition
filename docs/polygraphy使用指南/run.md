# polygraphy run - 跨框架推理比较

`polygraphy run` 是 Polygraphy 最核心的命令，用于在不同推理框架间比较模型输出，发现精度问题和性能差异。

## 🎯 主要功能

- **跨框架比较**: 同时在多个框架上运行模型并比较结果
- **逐层输出**: 支持标记所有中间层输出进行详细对比
- **自定义输入**: 支持加载自定义输入数据或使用随机数据
- **容差配置**: 灵活的误差容忍度设置
- **结果保存**: 可保存输入输出数据用于后续分析

## 📋 基本用法

### 简单跨框架比较
```bash
# ONNX Runtime vs TensorRT 比较
polygraphy run model.onnx --onnxrt --trt

# 包含原始 ONNX 框架
polygraphy run model.onnx --onnx --onnxrt --trt

# 指定特定版本
polygraphy run model.onnx --onnxrt --trt --trt-min-shapes input:[1,3,224,224]
```

### 逐层输出比较
```bash
# 标记所有层为输出
polygraphy run model.onnx --onnxrt --trt --mark-all

# 指定特定层
polygraphy run model.onnx --onnxrt --trt --onnx-outputs output1 output2
```

### 自定义输入数据
```bash
# 从 JSON 文件加载
polygraphy run model.onnx --onnxrt --trt --load-inputs inputs.json

# 使用数据生成脚本
polygraphy run model.onnx --onnxrt --trt --data-loader-script data_loader.py

# 指定输入形状范围
polygraphy run model.onnx --onnxrt --trt --input-shapes input:[1,3,224,224]
```

## ⚙️ 常用参数详解

### 框架选择参数
```bash
--onnxrt               # 使用 ONNX Runtime
--trt                  # 使用 TensorRT
--tf                   # 使用 TensorFlow
--pluginref            # 使用 Plugin CPU Reference
```

### 模型类型和加载参数
```bash
--model-type TYPE      # 指定输入模型类型
                      # frozen: TensorFlow 冻结图
                      # keras: Keras 模型
                      # ckpt: TensorFlow 检查点目录
                      # onnx: ONNX 模型
                      # engine: TensorRT 引擎
                      # uff: UFF 文件 [已弃用]
                      # trt-network-script: TensorRT 网络脚本
                      # caffe: Caffe prototxt [已弃用]
                      
--input-shapes         # 模型输入形状，格式: name:[shape]
                      # 例: --input-shapes image:[1,3,224,224] other:[10]
```

### ONNX 模型处理参数
```bash
--shape-inference     # 启用 ONNX 形状推理
--no-onnxruntime-shape-inference  # 禁用 ONNX Runtime 形状推理
--external-data-dir DIR           # 外部数据目录路径
--ignore-external-data           # 忽略外部数据，仅加载模型结构
--onnx-outputs OUTPUTS          # 指定 ONNX 输出张量
--onnx-exclude-outputs OUTPUTS  # 排除特定 ONNX 输出张量
--fp-to-fp16                    # 转换所有浮点张量为 FP16
--save-onnx PATH                # 保存 ONNX 模型路径
--save-external-data [PATH]     # 保存外部数据到文件
--external-data-size-threshold SIZE  # 外部数据大小阈值
```

### ONNX Runtime 配置参数
```bash
--providers PROVIDERS  # 执行提供程序，如 CPUExecutionProvider
                      # 例: --providers cuda cpu
```

### 输入数据参数
```bash
--load-inputs FILE     # 从 JSON 文件加载输入数据
--save-inputs FILE     # 保存生成的输入数据
--data-loader-script   # 自定义数据加载脚本
--data-loader-func-name NAME  # 数据加载函数名
--seed SEED           # 随机数种子，确保可重复性
--val-range RANGE     # 输入值范围，格式: [min,max] 或 input:[min,max]
--iterations NUM      # 推理迭代次数
--data-loader-backend-module  # 数据加载后端 (numpy/torch)
```

### 输出控制参数
```bash
--save-outputs FILE   # 保存所有框架的输出结果
--mark-all           # 将所有中间层标记为输出
--onnx-outputs       # 指定 ONNX 模型的输出层
--exclude-outputs    # 排除特定输出层
```

### TensorRT 特定参数
```bash
# 基础精度设置
--tf32              # 启用 TF32 精度
--fp16              # 启用 FP16 精度
--bf16              # 启用 BF16 精度
--fp8               # 启用 FP8 精度
--int8              # 启用 INT8 量化

# 形状配置
--trt-min-shapes SHAPES    # 最小输入形状 (动态形状)
--trt-opt-shapes SHAPES    # 优化输入形状 (最佳性能)
--trt-max-shapes SHAPES    # 最大输入形状 (动态形状)

# 量化配置
--calibration-cache PATH   # INT8 校准缓存路径
--calib-base-cls CLASS     # 校准基类 (如 IInt8MinMaxCalibrator)
--quantile QUANTILE        # IInt8LegacyCalibrator 分位数
--regression-cutoff CUTOFF # IInt8LegacyCalibrator 回归截止

# 引擎优化
--precision-constraints MODE  # 精度约束模式 (prefer/obey/none)
--sparse-weights           # 启用稀疏权重优化
--version-compatible       # 构建版本兼容引擎
--exclude-lean-runtime     # 排除精简运行时
--builder-optimization-level LEVEL  # 构建器优化级别
--hardware-compatibility-level LEVEL # 硬件兼容级别

# 内存和性能
--pool-limit POOL:SIZE     # 内存池限制
--max-aux-streams NUM      # 最大辅助流数量
--tactic-sources SOURCES   # 策略源 (cublas, cudnn 等)
--save-tactics PATH        # 保存策略重播文件
--load-tactics PATH        # 加载策略重播文件

# 缓存配置
--load-timing-cache PATH   # 加载时序缓存
--save-timing-cache PATH   # 保存时序缓存
--error-on-timing-cache-miss  # 时序缓存缺失时报错
--disable-compilation-cache   # 禁用编译缓存

# 高级功能
--weight-streaming         # 启用权重流
--weight-streaming-budget SIZE  # 权重流预算
--strongly-typed           # 强类型网络
--refittable               # 允许重新拟合权重
--strip-plan               # 构建时剥离可重新拟合权重

# DLA 支持
--use-dla                  # 使用 DLA 作为默认设备
--allow-gpu-fallback       # 允许 DLA 不支持的层回退到 GPU

# 插件和扩展
--plugins PATHS            # 加载插件库路径
--onnx-flags FLAGS         # ONNX 解析器标志
--plugin-instancenorm      # 强制使用插件 InstanceNorm

# 引擎文件操作
--save-engine PATH         # 保存 TensorRT 引擎
--load-runtime PATH        # 加载运行时

# 推理配置
--optimization-profile IDX # 推理时使用的优化配置文件索引
--allocation-strategy MODE # 激活内存分配策略 (static/profile/runtime)
```

### 比较和验证参数
```bash
# 基础比较配置
--validate               # 检查输出中的 NaN 和 Inf 值
--fail-fast             # 快速失败 (第一个失败后停止)
--compare {simple,indices}  # 比较函数类型
--compare-func-script SCRIPT  # 自定义比较函数脚本
--load-outputs PATHS    # 加载先前保存的输出结果
--no-shape-check        # 禁用形状检查

# 容差设置 (simple 比较函数)
--rtol RTOL             # 相对误差容忍度，支持按输出指定
                       # 例: --rtol 1e-5 output1:1e-4
--atol ATOL             # 绝对误差容忍度，支持按输出指定
                       # 例: --atol 1e-5 output1:1e-4
--check-error-stat STAT # 检查的误差统计量 (max/mean/median)
--infinities-compare-equal  # 匹配的 ±inf 值视为相等
--error-quantile QUANTILE   # 误差分位数比较

# 索引比较 (indices 比较函数)
--index-tolerance TOL   # 索引容忍度，支持按输出指定

# 结果可视化
--save-heatmaps DIR     # 保存误差热图
--show-heatmaps         # 显示误差热图
--save-error-metrics-plot DIR  # 保存误差指标图
--show-error-metrics-plot      # 显示误差指标图

# 后处理
--postprocess FUNC      # 输出后处理函数
                       # 例: --postprocess top-5 或 output1:top-3
```

### 推理配置参数
```bash
--warm-up NUM           # 预热运行次数
--use-subprocess        # 在独立子进程中运行
--save-outputs PATH     # 保存所有框架输出结果
```

### 日志和调试参数
```bash
# 日志级别控制
-v, --verbose           # 增加日志详细程度 (可多次使用)
-q, --quiet             # 减少日志详细程度 (可多次使用)
--verbosity LEVEL       # 指定日志详细级别 (INFO/VERBOSE/WARNING 等)
--silent                # 禁用所有输出

# 日志格式和输出
--log-format FORMAT     # 日志格式选项
                       # timestamp: 包含时间戳
                       # line-info: 包含文件和行号
                       # no-colors: 禁用颜色
--log-file PATH         # 将日志输出到文件

# 脚本生成和调试
--gen-script PATH       # 生成等效的 Python 脚本而不执行
                       # 用于理解和调试 polygraphy run 的行为
```

### TensorFlow 相关参数 (可选)
```bash
# TensorFlow 模型加载
--ckpt CHECKPOINT       # 检查点名称 (不含扩展名)
--tf-outputs OUTPUTS    # TensorFlow 输出张量名称
--save-pb PATH          # 保存 TensorFlow 冻结图
--freeze-graph          # 尝试冻结图

# TensorFlow 会话配置
--gpu-memory-fraction FRAC  # GPU 内存使用比例
--allow-growth          # 允许 GPU 内存动态增长
--xla                   # 启用 XLA 加速

# TensorFlow-TensorRT 集成
--tftrt                 # 启用 TF-TRT 集成
--minimum-segment-size SIZE  # 转换为 TensorRT 的最小段长度
--dynamic-op            # 启用动态模式 (运行时构建引擎)
```

## 💡 实用示例

### 1. 基础精度验证
```bash
# 简单的 ONNX Runtime vs TensorRT 比较
polygraphy run resnet50.onnx --onnxrt --trt --workspace 1G

# 查看详细日志
polygraphy run resnet50.onnx --onnxrt --trt --verbose
```

### 2. 动态形状模型比较
```bash
# 动态批次大小
polygraphy run model.onnx --onnxrt --trt \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[4,3,224,224] \
  --trt-max-shapes input:[8,3,224,224] \
  --input-shapes input:[4,3,224,224]
```

### 3. INT8 量化精度对比
```bash
# INT8 vs FP32 比较
polygraphy run model.onnx --onnxrt --trt --int8 \
  --calibration-cache calib.cache \
  --save-outputs int8_outputs.json
```

### 4. 自定义输入数据
```bash
# 使用真实数据
polygraphy run model.onnx --onnxrt --trt \
  --load-inputs real_data.json \
  --save-outputs results.json
```

### 5. 逐层精度分析
```bash
# 标记所有层输出
polygraphy run model.onnx --onnxrt --trt --onnx-outputs mark\ all \
  --save-outputs layer_outputs.json

# 仅比较特定层
polygraphy run model.onnx --onnxrt --trt \
  --onnx-outputs conv1_output relu1_output \
  --onnx-exclude-outputs final_output
```

### 6. 高级 TensorRT 优化
```bash
# 多精度对比测试
polygraphy run model.onnx --onnxrt --trt --fp16 \
  --rtol 1e-3 --atol 1e-3 \
  --builder-optimization-level 5 \
  --save-engine optimized_fp16.engine

# 权重流和内存优化
polygraphy run model.onnx --trt --strongly-typed --weight-streaming \
  --weight-streaming-budget 1G \
  --pool-limit workspace:2G \
  --max-aux-streams 4
```

### 7. 量化精度验证
```bash
# INT8 量化完整流程
polygraphy run model.onnx --onnxrt --trt --int8 \
  --data-loader-script calibration_data.py \
  --calibration-cache int8.cache \
  --calib-base-cls IInt8MinMaxCalibrator \
  --rtol 5e-2 --atol 1e-2
```

### 8. 调试和分析
```bash
# 生成调试脚本
polygraphy run model.onnx --onnxrt --trt \
  --gen-script debug_comparison.py

# 详细日志调试
polygraphy run model.onnx --onnxrt --trt \
  --verbose --verbose \
  --log-format timestamp line-info \
  --log-file debug.log

# 误差分析和可视化
polygraphy run model.onnx --onnxrt --trt \
  --validate --fail-fast \
  --save-heatmaps error_analysis/ \
  --save-error-metrics-plot plots/ \
  --check-error-stat max mean
```

### 9. 批量模型测试
```bash
# 创建批量测试脚本
#!/bin/bash
for precision in fp32 fp16 int8; do
    echo "Testing with $precision precision"
    polygraphy run model.onnx --onnxrt --trt --$precision \
      --save-outputs "results/model_${precision}.json" \
      --rtol 1e-3 --atol 1e-3
done

# 比较不同精度结果
polygraphy run model.onnx \
  --load-outputs results/model_fp32.json results/model_fp16.json \
  --compare simple --rtol 1e-2
```

### 10. 复杂输入数据场景
```bash
# 多输入模型
polygraphy run multi_input_model.onnx --onnxrt --trt \
  --input-shapes image:[1,3,224,224] mask:[1,1,224,224] \
  --val-range image:[0,1] mask:[0,1] \
  --data-loader-backend-module torch

# 动态形状完整测试
polygraphy run dynamic_model.onnx --onnxrt --trt \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[4,3,224,224] \
  --trt-max-shapes input:[8,3,224,224] \
  --input-shapes input:[2,3,224,224] input:[6,3,224,224] \
  --iterations 10
```

## 🔧 数据加载器脚本

### 基本数据加载器示例
```python
# data_loader.py
import numpy as np

def load_data():
    """
    生成器函数，产生输入数据
    返回字典形式: {"input_name": numpy_array}
    """
    for i in range(10):  # 生成10组数据
        yield {
            "input": np.random.randn(1, 3, 224, 224).astype(np.float32)
        }

# 使用方式
# polygraphy run model.onnx --onnxrt --trt --data-loader-script data_loader.py
```

### 真实数据加载器
```python
# real_data_loader.py
import cv2
import numpy as np
import os

def load_data():
    """加载真实图像数据"""
    image_dir = "test_images"
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image = image.transpose(2, 0, 1)  # HWC -> CHW
            image = image[np.newaxis, :] / 255.0  # 归一化
            
            yield {"input": image.astype(np.float32)}
```

### 多输入数据加载器
```python
# multi_input_loader.py
import numpy as np
import torch

def load_data():
    """多输入模型数据加载器"""
    batch_size = 1
    seq_len = 128
    vocab_size = 30522
    
    for i in range(10):  # 生成10批数据
        # 生成文本输入 ID
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # 生成注意力掩码
        attention_mask = np.ones((batch_size, seq_len))
        
        # 生成位置编码
        position_ids = np.arange(seq_len).reshape(1, -1)
        position_ids = np.tile(position_ids, (batch_size, 1))
        
        yield {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64)
        }
```

### INT8 校准数据加载器
```python
# calibration_data.py
import cv2
import numpy as np
import os
from pathlib import Path

def load_data():
    """INT8 校准数据加载器"""
    calib_images_dir = Path("calibration_images")
    image_files = list(calib_images_dir.glob("*.jpg")) + list(calib_images_dir.glob("*.png"))
    
    # 限制校准数据数量 (通常 100-1000 张图片足够)
    image_files = image_files[:500]
    
    for img_path in image_files:
        # 加载和预处理图像
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 标准化 (ImageNet 均值和标准差)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # 转换为 NCHW 格式
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, :]
        
        yield {"input": image.astype(np.float32)}
```

### 动态形状数据加载器
```python
# dynamic_shape_loader.py
import numpy as np

def load_data():
    """动态形状数据加载器"""
    # 定义不同的批次大小和序列长度
    batch_sizes = [1, 2, 4, 8]
    seq_lengths = [64, 128, 256, 512]
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # 生成随机输入数据
            input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            
            yield {"input": input_data}
            
            # 每种配置只生成一次
            break
```

## 📊 结果分析

### 输出格式理解
```json
{
  "inference_results": {
    "onnxrt-runner": {
      "output": [数组数据]
    },
    "trt-runner": {
      "output": [数组数据]  
    }
  },
  "comparison_results": {
    "output": {
      "max_error": 0.001,
      "mean_error": 0.0001,
      "passed": true
    }
  }
}
```

### 误差统计解读
- **max_error**: 最大绝对误差
- **mean_error**: 平均误差
- **passed**: 是否通过容差检查
- **error_distribution**: 误差分布统计

## ⚠️ 常见问题

### 1. 内存不足
```bash
# 减少批次大小
polygraphy run model.onnx --onnxrt --trt --input-shapes input:[1,3,224,224]

# 增加工作空间
polygraphy run model.onnx --onnxrt --trt --workspace 2G
```

### 2. 精度不匹配
```bash
# 调整容差
polygraphy run model.onnx --onnxrt --trt --rtol 1e-3 --atol 1e-3

# 使用 FP32 精度
polygraphy run model.onnx --onnxrt --trt --tf32
```

### 3. 动态形状问题
```bash
# 明确指定所有形状参数
polygraphy run model.onnx --onnxrt --trt \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[1,3,224,224] \
  --trt-max-shapes input:[1,3,224,224]
```

## 🚀 高级用法

### 1. 批量测试
```bash
# 创建测试脚本
for model in models/*.onnx; do
    echo "Testing $model"
    polygraphy run "$model" --onnxrt --trt --save-outputs "results/$(basename $model).json"
done
```

### 2. 性能基准测试
```bash
# 启用性能测量
polygraphy run model.onnx --onnxrt --trt --warm-up-runs 10 --timing-cache timing.cache
```

### 3. 调试模式  
```bash
# 详细调试信息
polygraphy run model.onnx --onnxrt --trt -vv --log-file debug.log
```

## 📈 最佳实践和性能优化

### 内存优化建议
```bash
# 大模型内存优化
polygraphy run large_model.onnx --onnxrt --trt \
  --pool-limit workspace:4G dla_local_dram:1G \
  --allocation-strategy runtime \
  --use-subprocess

# 权重流优化 (适用于超大模型)
polygraphy run huge_model.onnx --trt \
  --weight-streaming --weight-streaming-budget 50% \
  --strongly-typed --refittable
```

### 精度优化策略
```bash
# 渐进式精度测试
# 1. 首先测试 FP32
polygraphy run model.onnx --onnxrt --trt --save-outputs fp32_baseline.json

# 2. 测试 FP16 并与 FP32 比较
polygraphy run model.onnx --onnxrt --trt --fp16 \
  --load-outputs fp32_baseline.json \
  --rtol 1e-2 --atol 1e-3

# 3. 测试 INT8 并调整容差
polygraphy run model.onnx --onnxrt --trt --int8 \
  --calibration-cache calibration.cache \
  --load-outputs fp32_baseline.json \
  --rtol 5e-2 --atol 1e-2
```

### 性能基准测试
```bash
# 完整性能测试套件
polygraphy run model.onnx --onnxrt --trt --fp16 \
  --warm-up 10 --iterations 100 \
  --save-timing-cache timing.cache \
  --builder-optimization-level 5 \
  --tactic-sources cublas cudnn \
  --max-aux-streams 4
```

## 📚 相关文档

- [convert - 模型转换](./convert.md) - 模型格式转换
- [debug - 调试工具](./debug.md) - 进一步调试失败案例
- [inspect - 模型分析](./inspect.md) - 分析模型结构

---

*`polygraphy run` 是发现和调试推理问题的第一步，掌握其用法对于模型部署至关重要。*