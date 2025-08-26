# polygraphy inspect - 模型结构分析

`polygraphy inspect` 提供强大的模型分析功能，帮助理解模型结构、调试问题并验证模型属性。支持多种模型格式和数据分析。

## 🎯 主要功能

- **模型结构分析**: 查看层信息、输入输出形状、参数统计
- **数据检查**: 验证输入输出数据格式和取值范围
- **差异对比**: 比较不同模型版本的差异
- **性能分析**: 估算模型复杂度和内存占用
- **兼容性检查**: 验证模型在不同框架的兼容性

## 📋 子命令概览

| 子命令 | 功能 | 典型用法 |
|--------|------|----------|
| `model` | 模型结构分析 | `polygraphy inspect model model.onnx` |
| `data` | 数据文件检查 | `polygraphy inspect data inputs.json` |
| `tactics` | 策略重播文件检查 | `polygraphy inspect tactics replay.json` |
| `capability` | TensorRT兼容性检查 | `polygraphy inspect capability model.onnx` |
| `diff-tactics` | 策略差异分析 | `polygraphy inspect diff-tactics --good good/ --bad bad/` |
| `sparsity` | 稀疏性模式检查 | `polygraphy inspect sparsity model.onnx` |

## 🔧 通用日志参数

所有子命令都支持以下日志控制参数：

```bash
# 日志级别控制
-v, --verbose          # 增加日志详细程度 (可多次使用)
-q, --quiet            # 减少日志详细程度 (可多次使用)
--verbosity LEVEL      # 指定详细级别，支持路径级控制
--silent               # 禁用所有输出

# 日志格式和输出
--log-format FORMAT    # 日志格式: timestamp, line-info, no-colors
--log-file PATH        # 将日志输出到指定文件
```

## 🔍 inspect model - 模型结构分析

### 基本语法
```bash
polygraphy inspect model [options] model_file
```

### 模型文件参数
```bash
model_file                          # 模型文件路径

# 模型类型指定
--model-type {frozen,keras,ckpt,onnx,engine,uff,trt-network-script,caffe}
                                   # 指定模型类型
                                   # frozen: TensorFlow 冻结图
                                   # keras: Keras 模型  
                                   # ckpt: TensorFlow 检查点目录
                                   # onnx: ONNX 模型
                                   # engine: TensorRT 引擎
                                   # uff: UFF 文件 [已弃用]
                                   # trt-network-script: TensorRT 网络脚本
                                   # caffe: Caffe prototxt [已弃用]
```

### 显示控制参数
```bash
--convert-to {trt}                  # 转换为指定格式后再显示
--show {layers,attrs,weights}       # 控制显示内容
                                   # layers: 显示层信息 (名称、操作、输入输出)
                                   # attrs: 显示层属性 (需启用 layers)
                                   # weights: 显示权重信息
--list-unbounded-dds               # 列出无界数据相关形状(DDS)张量
--combine-tensor-info PATH         # 合并张量 JSON 文件信息
                                   # 仅支持 engine 类型和 layers 显示
```

### TensorFlow 模型加载参数
```bash
--ckpt CKPT                        # 检查点名称 (不含扩展名)
--freeze-graph                     # 尝试冻结图
```

### ONNX 形状推理参数
```bash
--shape-inference                  # 启用 ONNX 形状推理
--no-onnxruntime-shape-inference   # 禁用 ONNX Runtime 形状推理
```

### ONNX 模型加载参数
```bash
--external-data-dir DIR            # 外部数据目录路径
--ignore-external-data             # 忽略外部数据，仅加载模型结构
--fp-to-fp16                       # 转换所有浮点张量为 FP16
```

### TensorRT 插件加载参数
```bash
--plugins PLUGINS                  # 插件库路径
```

### TensorRT 网络加载参数
```bash
--layer-precisions PRECISIONS      # 每层计算精度
                                   # 格式: layer_name:precision
--tensor-dtypes DTYPES             # 网络 I/O 张量数据类型
                                   # 格式: tensor_name:datatype
--trt-network-func-name NAME       # 网络脚本函数名 [已弃用]
--trt-network-postprocess-script SCRIPT # 网络后处理脚本
--strongly-typed                   # 标记网络为强类型
--mark-debug TENSORS               # 标记调试张量
--mark-unfused-tensors-as-debug-tensors # 标记未融合张量为调试张量
```

### TensorRT 引擎参数
```bash
--save-timing-cache PATH           # 保存策略时序缓存
--load-runtime PATH                # 加载运行时 (版本兼容引擎)
```

### ONNX-TRT 解析器标志
```bash
--onnx-flags FLAGS                 # ONNX 解析器标志
--plugin-instancenorm              # 强制使用插件 InstanceNorm
```

### 基本用法示例
```bash
# 基础模型信息
polygraphy inspect model model.onnx

# 显示层和权重信息
polygraphy inspect model model.onnx --show layers weights

# 显示所有信息
polygraphy inspect model model.onnx --show layers attrs weights --list-unbounded-dds

# TensorRT 引擎分析
polygraphy inspect model model.engine --show layers weights

# 转换后分析
polygraphy inspect model model.onnx --convert-to trt --show layers
```

## 📊 inspect data - 数据文件检查

显示从 Polygraphy 的 Comparator.run() 保存的推理输入输出信息，例如通过 `--save-outputs` 或 `--save-inputs` 保存的数据。

### 基本语法
```bash
polygraphy inspect data [options] path
```

### 位置参数
```bash
path                        # 包含 Polygraphy 输入或输出数据的文件路径
```

### 显示控制参数
```bash
-a, --all                   # 显示所有迭代的信息，而不仅是第一个
-s, --show-values           # 显示张量值而不仅仅是元数据
--histogram                 # 显示值分布直方图
-n NUM_ITEMS, --num-items NUM_ITEMS
                           # 显示每个维度开始和结尾的值数量
                           # 使用 -1 显示所有元素，默认为 3
--line-width LINE_WIDTH     # 显示数组时每行的字符数
                           # 使用 -1 仅在维度端点插入换行，默认为 75
```

### 基本用法示例
```bash
# 检查推理输入数据文件
polygraphy inspect data inputs.json

# 检查输出结果文件并显示值
polygraphy inspect data outputs.json --show-values

# 显示所有迭代的信息
polygraphy inspect data results.json --all

# 显示值分布直方图
polygraphy inspect data data.json --histogram

# 自定义显示格式
polygraphy inspect data data.json --show-values --num-items 5 --line-width 100

# 显示完整数组内容
polygraphy inspect data small_tensor.json --show-values --num-items -1 --line-width -1
```

## 📋 inspect tactics - 策略重播文件检查

显示 Polygraphy 策略重播文件的内容，例如通过 `--save-tactics` 生成的文件，以人类可读的格式显示。

### 基本语法
```bash
polygraphy inspect tactics [options] tactic_replay
```

### 位置参数
```bash
tactic_replay               # 策略重播文件路径
```

### 基本用法示例
```bash
# 检查策略重播文件
polygraphy inspect tactics replay.json

# 详细日志输出
polygraphy inspect tactics replay.json --verbose

# 保存输出到文件
polygraphy inspect tactics replay.json > tactics_analysis.txt
```

## ⚙️ inspect capability - TensorRT兼容性检查

确定 TensorRT 运行 ONNX 图的能力。图将被分区为支持和不支持的子图，或仅根据静态检查错误进行分析。

### 基本语法
```bash
polygraphy inspect capability [options] model_file
```

### 位置参数
```bash
model_file                          # 模型文件路径
```

### 检查选项
```bash
--with-partitioning                 # 是否在解析失败的节点上对模型图进行分区
```

### ONNX 形状推理参数
```bash
--shape-inference                   # 启用 ONNX 形状推理
--no-onnxruntime-shape-inference    # 禁用 ONNX Runtime 形状推理
```

### ONNX 模型加载参数
```bash
--external-data-dir DIR             # 外部数据目录路径
--ignore-external-data              # 忽略外部数据，仅加载模型结构
--fp-to-fp16                        # 转换所有浮点张量为 FP16
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX    # 保存 ONNX 模型的目录路径
--save-external-data [PATH]         # 保存外部权重数据到文件
--external-data-size-threshold SIZE # 外部数据大小阈值 (字节)
--no-save-all-tensors-to-one-file   # 不将所有张量保存到一个文件
```

### 基本用法示例
```bash
# 检查 ONNX 模型的 TensorRT 兼容性
polygraphy inspect capability model.onnx

# 启用图分区分析
polygraphy inspect capability model.onnx --with-partitioning

# 详细兼容性报告
polygraphy inspect capability model.onnx --with-partitioning --verbose

# 启用形状推理进行兼容性检查
polygraphy inspect capability model.onnx --shape-inference --with-partitioning

# 保存支持的子图
polygraphy inspect capability model.onnx --with-partitioning -o supported_subgraphs/
```

## 🔍 inspect diff-tactics - 策略差异分析

根据好坏 Polygraphy 策略重播文件集合，确定潜在的坏 TensorRT 策略，例如通过 `--save-tactics` 保存的文件。

### 基本语法
```bash
polygraphy inspect diff-tactics [options]
```

### 策略文件参数
```bash
--dir DIR                           # 包含好坏策略重播文件的目录
                                   # 默认搜索名为 'good' 和 'bad' 的子目录
--good GOOD                         # 包含好策略重播文件的目录或单个文件
--bad BAD                           # 包含坏策略重播文件的目录或单个文件
```

### 基本用法示例
```bash
# 从默认目录结构分析策略差异
polygraphy inspect diff-tactics --dir tactics_data/

# 指定好坏策略文件目录
polygraphy inspect diff-tactics --good good_tactics/ --bad bad_tactics/

# 指定单个策略文件进行比较
polygraphy inspect diff-tactics --good good_replay.json --bad bad_replay.json

# 详细分析报告
polygraphy inspect diff-tactics --good good/ --bad bad/ --verbose
```

## 📊 inspect sparsity - 稀疏性模式检查

[实验性功能] 显示 ONNX 模型中每个权重张量是否遵循 2:4 结构化稀疏性模式的信息。

### 基本语法
```bash
polygraphy inspect sparsity [options] model_file
```

### 位置参数
```bash
model_file                          # 模型文件路径
```

### ONNX 模型加载参数
```bash
--external-data-dir DIR             # 外部数据目录路径
--ignore-external-data              # 忽略外部数据，仅加载模型结构
--fp-to-fp16                        # 转换所有浮点张量为 FP16
```

### 基本用法示例
```bash
# 检查模型的稀疏性模式
polygraphy inspect sparsity model.onnx

# 详细稀疏性分析
polygraphy inspect sparsity model.onnx --verbose

# 忽略外部数据检查稀疏性
polygraphy inspect sparsity model.onnx --ignore-external-data
```

## 💡 实用示例

### 1. 新模型快速分析
```bash
# 完整模型分析流水线
polygraphy inspect model model.onnx --show layers attrs weights --list-unbounded-dds

# 检查 TensorRT 兼容性
polygraphy inspect capability model.onnx --with-partitioning --verbose

# 检查稀疏性模式
polygraphy inspect sparsity model.onnx
```

### 2. 调试模型转换失败
```bash
# 分析原始模型结构
polygraphy inspect model problematic.onnx --show layers --list-unbounded-dds

# 检查 TensorRT 兼容性问题
polygraphy inspect capability problematic.onnx --with-partitioning --verbose

# 转换后再分析
polygraphy inspect model problematic.onnx --convert-to trt --show layers
```

### 3. 推理结果调试工作流
```bash
# 1. 分析推理输入数据
polygraphy inspect data inputs.json --show-values --histogram

# 2. 分析推理结果
polygraphy inspect data outputs.json --all --show-values

# 3. 检查策略重播文件
polygraphy inspect tactics good_tactics.json
polygraphy inspect tactics bad_tactics.json

# 4. 分析策略差异
polygraphy inspect diff-tactics --good good_tactics.json --bad bad_tactics.json
```

### 4. 动态形状模型分析
```bash
# 查看动态形状信息
polygraphy inspect model dynamic_model.onnx --list-unbounded-dds --show layers attrs

# 分析 TensorRT 引擎的动态形状
polygraphy inspect model dynamic.engine --show layers attrs weights
```

### 5. 大模型内存估算
```bash
# 分析模型内存需求
polygraphy inspect model large_model.onnx --show layers weights | grep -i "size\|memory"

# 检查 TensorRT 兼容性和内存要求
polygraphy inspect capability large_model.onnx --with-partitioning --verbose
```

## 📈 输出解读指南

### 模型结构输出示例
```
Model: model.onnx
    Name: resnet50 | ONNX Opset: 11

    ---- 1 Graph Input(s) ----
    {input} [dtype=float32, shape=(1, 3, 224, 224)]

    ---- 1 Graph Output(s) ----  
    {output} [dtype=float32, shape=(1, 1000)]

    ---- 161 Initializer(s) ----
    Conv_0.weight [dtype=float32, shape=(64, 3, 7, 7)] | Stats: mean=0.001, std=0.045, min=-0.123, max=0.098
    ...

    ---- 174 Node(s) ----
    Node 0    | [Op: Conv]
        {input} -> {Conv_0}
        weight: Conv_0.weight [shape=(64, 3, 7, 7)]
        bias: Conv_0.bias [shape=(64,)]
    ...
```

### 关键信息解读
- **Graph Input/Output**: 模型输入输出张量的名称、类型、形状
- **Initializer**: 模型权重参数，包含统计信息
- **Nodes**: 计算节点，显示操作类型和连接关系
- **Stats**: 权重统计：均值、标准差、最值

### 数据检查输出示例
```
Data: inputs.json
    
    ---- Input: input ----
    dtype: float32 | shape: (1, 3, 224, 224)
    Stats: mean=0.485, std=0.229, min=0.0, max=1.0
    
    Values (first 10):
    [0.485, 0.456, 0.406, ...]
```

## 🔧 自动化脚本

### 批量模型分析
```bash
#!/bin/bash
# batch_inspect.sh

models_dir="models"
reports_dir="inspection_reports"
mkdir -p "$reports_dir"

for model in "$models_dir"/*.onnx; do
    model_name=$(basename "$model" .onnx)
    echo "分析模型: $model_name"
    
    # 基本分析
    polygraphy inspect model "$model" --show layers attrs weights \
      > "$reports_dir/${model_name}_analysis.txt"
    
    # 检查 TensorRT 兼容性
    polygraphy inspect model "$model" --convert-to=trt \
      > "$reports_dir/${model_name}_trt_compat.log" 2>&1
    
    echo "完成: $model_name"
done
```

### 模型对比报告生成
```bash
#!/bin/bash
# compare_models.sh

model1=$1
model2=$2
output_dir="comparison_report"

mkdir -p "$output_dir"

echo "比较模型: $(basename $model1) vs $(basename $model2)"

# 模型1分析
polygraphy inspect model "$model1" --show layers attrs weights \
  > "$output_dir/model1_analysis.txt"

# 模型2分析  
polygraphy inspect model "$model2" --show layers attrs weights \
  > "$output_dir/model2_analysis.txt"

# 注意: polygraphy inspect 不支持直接模型对比
# 需要使用其他工具或手动比较分析结果

echo "报告生成完成: $output_dir/"
```

### Python API 使用
```python
# model_inspector.py
from polygraphy.tools.args import ModelArgs
from polygraphy.tools.inspect.subtool import InspectModel
import json

def analyze_model(model_path):
    """使用 Python API 分析模型"""
    # 创建模型参数
    model_args = ModelArgs()
    model_args.path = model_path
    
    # 创建检查工具
    inspector = InspectModel()
    
    # 设置参数
    inspector.mode = "full"
    inspector.show_weights = True
    
    # 执行分析
    result = inspector.run(model_args)
    
    return result

# 使用示例
if __name__ == "__main__":
    model_info = analyze_model("model.onnx")
    print(json.dumps(model_info, indent=2))
```

## ⚠️ 注意事项

### 1. 大模型分析
```bash
# 大模型可能需要更多内存和时间，先显示基础信息
polygraphy inspect model large_model.onnx --show layers

# 如果内存不足，避免显示权重
polygraphy inspect model large_model.onnx --show layers attrs  # 不要加 weights
```

### 2. 动态形状模型
```bash
# 动态形状模型需要特别注意
polygraphy inspect model dynamic.onnx --list-unbounded-dds --verbose
```

### 3. 加密或受保护的模型
```bash
# 某些模型可能有访问限制
polygraphy inspect model protected.onnx --verbose  # 查看详细错误信息
```

## 🚀 高级用法

### 1. 自定义输出格式
```python
# custom_inspector.py
import json
from polygraphy.tools.inspect.model import inspect_model

def custom_model_analysis(model_path):
    """自定义模型分析输出"""
    analysis = inspect_model(model_path, mode="full")
    
    # 提取关键信息
    summary = {
        "model_name": analysis.get("name", "unknown"),
        "input_shapes": {inp.name: inp.shape for inp in analysis.inputs},
        "output_shapes": {out.name: out.shape for out in analysis.outputs},
        "total_parameters": sum(w.size for w in analysis.weights),
        "model_size_mb": sum(w.nbytes for w in analysis.weights) / (1024**2)
    }
    
    return summary

# 批量分析
models = ["model1.onnx", "model2.onnx", "model3.onnx"]
results = {}

for model in models:
    results[model] = custom_model_analysis(model)

# 保存结果
with open("model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 2. 持续集成中的模型验证
```yaml
# .github/workflows/model_validation.yml
name: Model Validation

on: [push, pull_request]

jobs:
  validate-models:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Install Polygraphy
      run: pip install polygraphy
      
    - name: Validate Models
      run: |
        for model in models/*.onnx; do
          echo "Validating $model"
          polygraphy inspect model "$model" --show layers || exit 1
          
          # 检查 TensorRT 兼容性
          polygraphy inspect capability "$model" --with-partitioning --verbose || echo "TRT compatibility issue: $model"
        done
```

## 📚 相关文档

- [run - 跨框架比较](./run.md) - 使用分析结果优化运行参数
- [convert - 模型转换](./convert.md) - 基于分析结果调整转换策略  
- [surgeon - 模型修改](./surgeon.md) - 根据分析结果修改模型结构

---

*`polygraphy inspect` 是理解和调试模型的第一步，详细的分析有助于后续的优化和部署决策。*