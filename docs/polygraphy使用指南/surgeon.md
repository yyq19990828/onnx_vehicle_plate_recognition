# polygraphy surgeon - ONNX 模型修改

`polygraphy surgeon` 是专门用于修改 ONNX 模型的工具集，提供子图提取、模型清理、节点插入、权重修剪等功能，帮助优化模型结构和解决兼容性问题。

## 🎯 主要功能

- **子图提取**: 从 ONNX 模型中提取特定子图用于调试或重用
- **模型清理和优化**: 清理、优化和更改输入形状
- **节点插入**: 插入单个节点并替换现有子图
- **权重修剪**: 修剪权重以遵循 2:4 结构化稀疏性模式
- **权重剥离**: 从模型中剥离或重建权重

## 📋 基本语法

```bash
polygraphy surgeon [-h] [-v] [-q] [--verbosity VERBOSITY [VERBOSITY ...]] [--silent]
                  [--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]]
                  [--log-file LOG_FILE]
                  {extract,sanitize,insert,prune,weight-strip,weight-reconstruct} ...
```

## 🔧 通用日志参数

所有子命令都支持以下日志控制参数：

```bash
-h, --help                     # 显示帮助信息并退出

# 日志级别控制
-v, --verbose                  # 增加日志详细程度 (可多次使用)
-q, --quiet                    # 减少日志详细程度 (可多次使用)
--verbosity VERBOSITY [VERBOSITY ...]
                               # 指定详细级别，支持路径级控制
                               # 格式: <path>:<verbosity>
                               # 例如: --verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE
--silent                       # 禁用所有输出

# 日志格式和输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
                               # 日志格式控制:
                               # timestamp: 包含时间戳
                               # line-info: 包含文件和行号
                               # no-colors: 禁用颜色
--log-file LOG_FILE            # 将日志输出到指定文件
```

## 📊 子命令概览

| 子命令 | 功能 | 典型用法 |
|--------|------|----------|
| `extract` | 从 ONNX 模型中提取基于指定输入输出的子图 | `polygraphy surgeon extract model.onnx --inputs input1 --outputs output1 -o subgraph.onnx` |
| `sanitize` | 清理、优化和/或更改 ONNX 模型中的输入形状 | `polygraphy surgeon sanitize model.onnx --fold-constants -o clean.onnx` |
| `insert` | [实验性] 插入单个节点到 ONNX 模型中 | `polygraphy surgeon insert model.onnx --inputs input1 --outputs output1 --op Relu -o modified.onnx` |
| `prune` | [实验性] 修剪权重遵循 2:4 结构化稀疏性模式 | `polygraphy surgeon prune model.onnx -o pruned.onnx` |
| `weight-strip` | 从提供的 ONNX 模型中剥离权重 | `polygraphy surgeon weight-strip model.onnx -o stripped.onnx` |
| `weight-reconstruct` | 在剥离的 ONNX 模型中重建代理权重 | `polygraphy surgeon weight-reconstruct stripped.onnx -o reconstructed.onnx` |

## 🔍 extract - 子图提取

从 ONNX 模型中基于指定的输入和输出提取子图。

### 基本语法
```bash
polygraphy surgeon extract [-h] [日志参数] [模型参数] [数据加载器参数] [ONNX参数] [保存参数] 
                          [--inputs INPUT_META [INPUT_META ...]] [--outputs OUTPUT_META [OUTPUT_META ...]] 
                          model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### 子图定义参数
```bash
--inputs INPUT_META [INPUT_META ...]
                               # 子图的输入元数据 (名称、形状和数据类型)
                               # 使用 'auto' 让 extract 自动确定
                               # 格式: --inputs <name>:<shape>:<dtype>
                               # 例如: --inputs input0:[1,3,224,224]:float32 input1:auto:auto
                               # 如果省略，使用当前模型的输入
--outputs OUTPUT_META [OUTPUT_META ...]
                               # 子图的输出元数据 (名称和数据类型)
                               # 使用 'auto' 让 extract 自动确定
                               # 格式: --outputs <name>:<dtype>
                               # 例如: --outputs output0:float32 output1:auto
                               # 如果省略，使用当前模型的输出
```

### 模型参数
```bash
--model-input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --model-inputs INPUT_SHAPES [INPUT_SHAPES ...]
                               # 运行回退形状推理时生成数据使用的输入形状
                               # 格式: --model-input-shapes <name>:<shape>
                               # 例如: --model-input-shapes image:[1,3,224,224] other_input:[10]
```

### 数据加载器参数
```bash
--seed SEED                    # 随机输入的种子
--val-range VAL_RANGE [VAL_RANGE ...]
                               # 数据加载器中生成的值范围
                               # 格式: --val-range <input_name>:[min,max]
                               # 例如: --val-range [0,1] inp0:[2,50] inp1:[3.0,4.6]
--int-min INT_MIN              # [已弃用: 使用 --val-range] 随机整数输入的最小值
--int-max INT_MAX              # [已弃用: 使用 --val-range] 随机整数输入的最大值
--float-min FLOAT_MIN          # [已弃用: 使用 --val-range] 随机浮点输入的最小值
--float-max FLOAT_MAX          # [已弃用: 使用 --val-range] 随机浮点输入的最大值
--iterations NUM, --iters NUM  # 默认数据加载器应提供数据的推理迭代次数
--data-loader-backend-module {numpy,torch}
                               # 用于生成输入数组的模块，支持: numpy, torch
--load-inputs LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...], --load-input-data LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...]
                               # 加载输入的路径，文件应为 JSON 化的 List[Dict[str, numpy.ndarray]]
--data-loader-script DATA_LOADER_SCRIPT
                               # Python 脚本路径，定义加载输入数据的函数
                               # 格式: my_custom_script.py:my_func
--data-loader-func-name DATA_LOADER_FUNC_NAME
                               # [已弃用] 数据加载器脚本中加载数据的函数名称，默认为 load_data
```

### ONNX 形状推理参数
```bash
--shape-inference, --do-shape-inference
                               # 加载模型时启用 ONNX 形状推理
--force-fallback-shape-inference
                               # 强制使用 ONNX-Runtime 确定图中张量的元数据
                               # 这会导致动态维度变为静态
--no-onnxruntime-shape-inference
                               # 禁用使用 ONNX-Runtime 的形状推理实用程序
                               # 强制使用 onnx.shape_inference
```

### ONNX 模型加载参数
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                               # 包含模型外部数据的目录路径
--ignore-external-data         # 忽略外部数据，仅加载模型结构
--fp-to-fp16                   # 将 ONNX 模型中的所有浮点张量转换为 16 位精度
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                               # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值，超过此阈值的张量数据将存储在外部文件中
                               # 支持 K、M、G 后缀，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 基本用法示例
```bash
# 提取基本子图
polygraphy surgeon extract model.onnx --inputs input1 --outputs output1 -o subgraph.onnx

# 指定输入形状和数据类型
polygraphy surgeon extract model.onnx \
  --inputs input0:[1,3,224,224]:float32 \
  --outputs output0:float32 \
  -o typed_subgraph.onnx

# 自动确定输入输出元数据
polygraphy surgeon extract model.onnx \
  --inputs input1:auto:auto \
  --outputs output1:auto \
  -o auto_subgraph.onnx
```

## 🧹 sanitize - 模型清理

清理、优化和/或更改 ONNX 模型中的输入形状。

### 基本语法
```bash
polygraphy surgeon sanitize [-h] [日志参数] [模型参数] [数据加载器参数] [ONNX参数] [保存参数] [常量折叠参数] 
                           [--cleanup] [--toposort] model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### 基本清理参数
```bash
--cleanup                      # 在图上运行死层移除，如果设置了其他选项通常不需要
--toposort                     # 对图中的节点进行拓扑排序
```

### 模型参数
```bash
--override-input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --override-inputs INPUT_SHAPES [INPUT_SHAPES ...]
                               # 覆盖模型中给定输入的输入形状
                               # 格式: --override-input-shapes <name>:<shape>
                               # 例如: --override-input-shapes image:[1,3,224,224] other_input:[10]
```

### 数据加载器参数
```bash
--seed SEED                    # 随机输入的种子
--val-range VAL_RANGE [VAL_RANGE ...]
                               # 数据加载器中生成的值范围
--int-min INT_MIN              # [已弃用: 使用 --val-range] 随机整数输入的最小值
--int-max INT_MAX              # [已弃用: 使用 --val-range] 随机整数输入的最大值
--float-min FLOAT_MIN          # [已弃用: 使用 --val-range] 随机浮点输入的最小值
--float-max FLOAT_MAX          # [已弃用: 使用 --val-range] 随机浮点输入的最大值
--iterations NUM, --iters NUM  # 默认数据加载器应提供数据的推理迭代次数
--data-loader-backend-module {numpy,torch}
                               # 用于生成输入数组的模块
--load-inputs LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...], --load-input-data LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...]
                               # 加载输入的路径
--data-loader-script DATA_LOADER_SCRIPT
                               # Python 脚本路径，定义加载输入数据的函数
--data-loader-func-name DATA_LOADER_FUNC_NAME
                               # [已弃用] 数据加载器函数名称
```

### ONNX 形状推理参数
```bash
--no-shape-inference           # 加载模型时禁用 ONNX 形状推理
--force-fallback-shape-inference
                               # 强制使用 ONNX-Runtime 确定图中张量的元数据
--no-onnxruntime-shape-inference
                               # 禁用使用 ONNX-Runtime 的形状推理实用程序
```

### ONNX 模型加载参数
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                               # 包含模型外部数据的目录路径
--ignore-external-data         # 忽略外部数据，仅加载模型结构
--outputs ONNX_OUTPUTS [ONNX_OUTPUTS ...]
                               # 要标记为输出的 ONNX 张量名称
                               # 使用特殊值 'mark all' 表示所有张量都应用作输出
--exclude-outputs ONNX_EXCLUDE_OUTPUTS [ONNX_EXCLUDE_OUTPUTS ...]
                               # [实验性] 要取消标记为输出的 ONNX 输出名称
--fp-to-fp16                   # 将 ONNX 模型中的所有浮点张量转换为 16 位精度
--set-unbounded-dds-upper-bound UPPER_BOUNDS [UPPER_BOUNDS ...]
                               # 为具有无界 DDS(数据相关形状)的张量设置上界
                               # 格式: --set-unbounded-dds-upper-bound [<tensor_name>:]<upper_bound>
                               # 例如: --set-unbounded-dds-upper-bound 10000 tensor_a:5000 tensor_b:4000
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                               # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 常量折叠参数
```bash
--fold-constants               # 通过计算不依赖于运行时输入的子图来折叠图中的常量
--num-passes NUM_CONST_FOLD_PASSES, --num-const-fold-passes NUM_CONST_FOLD_PASSES
                               # 运行的常量折叠通道数，如果未指定则自动确定
--partitioning {basic,recursive}
                               # 控制常量折叠期间如何分区图:
                               # basic: 分区图使一部分的故障不影响其他部分
                               # recursive: 除了分区图外，还在需要时分区分区
--no-fold-shapes               # 禁用折叠 Shape 节点和在形状上操作的子图
--no-per-pass-shape-inference  # 禁用常量折叠通道之间的形状推理
--fold-size-threshold FOLD_SIZE_THRESHOLD
                               # 应用常量折叠的每张量最大大小阈值 (字节)
                               # 支持 K、M、G 后缀
```

### 基本用法示例
```bash
# 基础清理
polygraphy surgeon sanitize model.onnx --cleanup -o clean.onnx

# 常量折叠和拓扑排序
polygraphy surgeon sanitize model.onnx --fold-constants --toposort -o optimized.onnx

# 覆盖输入形状
polygraphy surgeon sanitize model.onnx \
  --override-input-shapes input:[1,3,224,224] \
  -o reshaped.onnx

# 高级常量折叠
polygraphy surgeon sanitize model.onnx \
  --fold-constants \
  --num-passes 3 \
  --partitioning recursive \
  --fold-size-threshold 16M \
  -o advanced_folded.onnx
```

## ➕ insert - 节点插入

[实验性] 将单个节点插入到 ONNX 模型中，具有指定的输入和输出。输入和输出之间的任何现有子图都会被替换。

### 基本语法
```bash
polygraphy surgeon insert [-h] [日志参数] [ONNX参数] [保存参数] 
                         --inputs INPUTS [INPUTS ...] --outputs OUTPUTS [OUTPUTS ...] --op OP
                         [--name NAME] [--attrs ATTRS [ATTRS ...]] 
                         model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### 插入节点参数
```bash
--inputs INPUTS [INPUTS ...]   # 新节点的输入张量名称，将保持顺序
                               # 格式: --inputs <name>
                               # 例如: --inputs name0 name1
--outputs OUTPUTS [OUTPUTS ...]
                               # 新节点的输出张量名称，将保持顺序
                               # 如果输出张量也被指定为输入，将为输出生成新张量
                               # 格式: --outputs <name>
                               # 例如: --outputs name0 name1
--op OP                        # 新节点使用的 ONNX 操作
--name NAME                    # 新节点使用的名称
--attrs ATTRS [ATTRS ...]      # 在新节点中设置的属性
                               # 格式: --attrs <name>=value
                               # 例如: --attrs axis=1 keepdims=1
                               # 支持类型: float, int, str 以及这些类型的列表
                               # 包含小数点的数字总是被解析为浮点数
                               # 带引号的值总是被解析为字符串
                               # 用括号括起来的值被解析为列表
```

### ONNX 形状推理参数
```bash
--shape-inference, --do-shape-inference
                               # 加载模型时启用 ONNX 形状推理
--no-onnxruntime-shape-inference
                               # 禁用使用 ONNX-Runtime 的形状推理实用程序
```

### ONNX 模型加载参数
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                               # 包含模型外部数据的目录路径
--ignore-external-data         # 忽略外部数据，仅加载模型结构
--fp-to-fp16                   # 将 ONNX 模型中的所有浮点张量转换为 16 位精度
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                               # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 基本用法示例
```bash
# 插入 Relu 节点
polygraphy surgeon insert model.onnx \
  --inputs intermediate_tensor \
  --outputs relu_output \
  --op Relu \
  --name debug_relu \
  -o modified.onnx

# 插入带属性的节点
polygraphy surgeon insert model.onnx \
  --inputs input_tensor \
  --outputs output_tensor \
  --op Transpose \
  --attrs perm=[0,2,1,3] \
  --name transpose_node \
  -o transposed.onnx
```

## 🔧 prune - 权重修剪

[实验性] 修剪模型的权重以遵循 2:4 结构化稀疏性模式，不考虑准确性。每四个权重值中，两个将被设置为零。

**注意:** 此工具用于帮助功能测试稀疏性。它几乎肯定会导致显著的准确性下降，因此不应在功能测试之外使用。

### 基本语法
```bash
polygraphy surgeon prune [-h] [日志参数] [ONNX参数] [保存参数] model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### ONNX 模型加载参数
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                               # 包含模型外部数据的目录路径
--ignore-external-data         # 忽略外部数据，仅加载模型结构
--fp-to-fp16                   # 将 ONNX 模型中的所有浮点张量转换为 16 位精度
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                               # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 基本用法示例
```bash
# 修剪模型权重为 2:4 稀疏性模式
polygraphy surgeon prune model.onnx -o pruned.onnx

# 详细日志修剪
polygraphy surgeon prune model.onnx -o pruned.onnx --verbose
```

## 🗂️ weight-strip - 权重剥离

从提供的 ONNX 模型中剥离权重。

### 基本语法
```bash
polygraphy surgeon weight-strip [-h] [日志参数] [ONNX参数] [保存参数] [--exclude-list EXCLUDE_LIST] model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### 权重剥离参数
```bash
--exclude-list EXCLUDE_LIST    # 包含要跳过的初始化器列表的文本文件路径
```

### ONNX 模型加载参数
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                               # 包含模型外部数据的目录路径
--ignore-external-data         # 忽略外部数据，仅加载模型结构
--fp-to-fp16                   # 将 ONNX 模型中的所有浮点张量转换为 16 位精度
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                               # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 基本用法示例
```bash
# 剥离所有权重
polygraphy surgeon weight-strip model.onnx -o stripped.onnx

# 剥离权重但排除指定的初始化器
polygraphy surgeon weight-strip model.onnx \
  --exclude-list exclude_weights.txt \
  -o selective_stripped.onnx
```

## 🔄 weight-reconstruct - 权重重建

在剥离的 ONNX 模型中重建代理权重。

### 基本语法
```bash
polygraphy surgeon weight-reconstruct [-h] [日志参数] [ONNX参数] [保存参数] model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### ONNX 模型加载参数
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                               # 包含模型外部数据的目录路径
--ignore-external-data         # 忽略外部数据，仅加载模型结构
--fp-to-fp16                   # 将 ONNX 模型中的所有浮点张量转换为 16 位精度
```

### ONNX 模型保存参数
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                               # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 基本用法示例
```bash
# 重建代理权重
polygraphy surgeon weight-reconstruct stripped.onnx -o reconstructed.onnx

# 详细日志重建
polygraphy surgeon weight-reconstruct stripped.onnx -o reconstructed.onnx --verbose
```

## 💡 实用示例

### 1. 调试模型准备
```bash
# 清理模型并提取子图
polygraphy surgeon sanitize problematic.onnx --fold-constants -o clean.onnx
polygraphy surgeon extract clean.onnx \
  --inputs input \
  --outputs conv1_output \
  -o debug_subgraph.onnx
```

### 2. 模型性能优化
```bash
# 完整优化流程
polygraphy surgeon sanitize model.onnx \
  --fold-constants \
  --num-passes 5 \
  --toposort \
  --cleanup \
  -o stage1.onnx

# 进一步优化
polygraphy surgeon sanitize stage1.onnx \
  --partitioning recursive \
  --fold-size-threshold 16M \
  -o optimized.onnx
```

### 3. 权重管理工作流
```bash
# 剥离权重用于分发
polygraphy surgeon weight-strip large_model.onnx -o lightweight.onnx

# 重建权重用于测试
polygraphy surgeon weight-reconstruct lightweight.onnx -o test_model.onnx

# 修剪权重进行稀疏性测试
polygraphy surgeon prune test_model.onnx -o sparse_test.onnx
```

### 4. 动态形状处理
```bash
# 固定输入形状
polygraphy surgeon sanitize dynamic_model.onnx \
  --override-input-shapes input:[1,3,224,224] \
  --fold-constants \
  -o fixed_shape.onnx

# 提取固定形状的子图
polygraphy surgeon extract fixed_shape.onnx \
  --inputs input:[1,3,224,224]:float32 \
  --outputs output:float32 \
  -o final_subgraph.onnx
```

### 5. 节点替换和插入
```bash
# 插入调试节点
polygraphy surgeon insert model.onnx \
  --inputs intermediate \
  --outputs debug_output \
  --op Identity \
  --name debug_identity \
  -o debug_model.onnx

# 插入激活函数
polygraphy surgeon insert model.onnx \
  --inputs conv_output \
  --outputs relu_output \
  --op Relu \
  --name inserted_relu \
  -o activated_model.onnx
```

## ⚠️ 注意事项

### 1. 大模型处理
```bash
# 大模型常量折叠可能消耗大量内存
polygraphy surgeon sanitize large_model.onnx \
  --fold-constants \
  --fold-size-threshold 512M \
  -o optimized.onnx
```

### 2. 实验性功能警告
```bash
# insert 和 prune 是实验性功能，谨慎使用
polygraphy surgeon prune model.onnx -o pruned.onnx --verbose

# prune 会导致准确性下降，仅用于功能测试
```

### 3. 权重剥离和重建
```bash
# 确保剥离和重建的一致性
polygraphy surgeon weight-strip model.onnx -o stripped.onnx
polygraphy surgeon weight-reconstruct stripped.onnx -o reconstructed.onnx
```

## 📚 相关文档

- [inspect - 模型分析](./inspect.md) - 分析修改后的模型结构
- [run - 跨框架比较](./run.md) - 验证修改后的模型精度
- [convert - 模型转换](./convert.md) - 将修改后的模型转换为其他格式

---

*`polygraphy surgeon` 是模型修改和优化的重要工具，合理使用可以显著提高模型的兼容性和性能。*