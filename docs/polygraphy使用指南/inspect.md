# polygraphy inspect - 模型结构分析

`polygraphy inspect` 提供强大的模型分析功能，帮助理解模型结构、调试问题并验证模型属性。支持多种模型格式和数据分析。

## 🎯 主要功能

- **模型结构分析**: 查看层信息、输入输出形状、参数统计
- **数据检查**: 验证输入输出数据格式和取值范围
- **策略分析**: 检查和比较 TensorRT 策略重播文件
- **兼容性检查**: 验证模型在 TensorRT 中的兼容性
- **稀疏性检查**: 分析模型的 2:4 结构化稀疏性模式

## 📋 基本语法

```bash
polygraphy inspect [-h] [-v] [-q] [--verbosity VERBOSITY [VERBOSITY ...]] [--silent]
                  [--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]]
                  [--log-file LOG_FILE]
                  {model,data,tactics,capability,diff-tactics,sparsity} ...
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
| `model` | 显示模型信息，包括输入输出和层属性 | `polygraphy inspect model model.onnx` |
| `data` | 显示从 Comparator.run() 保存的推理数据信息 | `polygraphy inspect data inputs.json` |
| `tactics` | 显示策略重播文件内容 | `polygraphy inspect tactics replay.json` |
| `capability` | 确定 TensorRT 运行 ONNX 图的能力 | `polygraphy inspect capability model.onnx` |
| `diff-tactics` | 确定潜在的坏 TensorRT 策略 | `polygraphy inspect diff-tactics --good good/ --bad bad/` |
| `sparsity` | [实验性] 显示权重张量的 2:4 结构化稀疏性模式 | `polygraphy inspect sparsity model.onnx` |

## 🔍 inspect model - 模型结构分析

显示模型信息，包括输入输出以及层和它们的属性。

### 基本语法
```bash
polygraphy inspect model [-h] [日志参数] [模型参数] [显示控制] [各种加载参数] model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### 显示控制参数
```bash
--convert-to {trt}, --display-as {trt}
                               # 在显示前尝试将模型转换为指定格式
--show {layers,attrs,weights} [{layers,attrs,weights} ...]
                               # 控制显示内容:
                               # layers: 显示基本层信息 (名称、操作、输入输出)
                               # attrs: 显示所有可用的每层属性 (需要启用 layers)
                               # weights: 显示模型中的所有权重
--list-unbounded-dds           # 列出所有具有无界数据相关形状(DDS)的张量
--combine-tensor-info COMBINE_TENSOR_INFO
                               # 设置张量 JSON 文件路径以合并信息到层的输入输出信息中
                               # 仅在 --model-type 为 "engine" 且 --show 包含 "layers" 时支持
```

### 模型类型参数
```bash
--model-type {frozen,keras,ckpt,onnx,engine,uff,trt-network-script,caffe}
                               # 输入模型的类型:
                               # frozen: TensorFlow 冻结图
                               # keras: Keras 模型
                               # ckpt: TensorFlow 检查点目录
                               # onnx: ONNX 模型
                               # engine: TensorRT 引擎
                               # uff: UFF 文件 [已弃用]
                               # trt-network-script: Python 脚本，定义 load_network 函数
                               # caffe: Caffe prototxt [已弃用]
```

### TensorFlow 模型加载参数
```bash
--ckpt CKPT                    # [实验性] 要加载的检查点名称
                               # 如果缺少 checkpoint 文件则必需
                               # 不应包含文件扩展名
--freeze-graph                 # [实验性] 尝试冻结图
```

### ONNX 形状推理参数
```bash
--shape-inference, --do-shape-inference
                               # 加载模型时启用 ONNX 形状推理
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

### TensorRT 插件加载参数
```bash
--plugins PLUGINS [PLUGINS ...]
                               # 要加载的插件库路径
```

### TensorRT 网络加载参数
```bash
--layer-precisions LAYER_PRECISIONS [LAYER_PRECISIONS ...]
                               # 每层使用的计算精度
                               # 格式: --layer-precisions <layer_name>:<layer_precision>
--tensor-dtypes TENSOR_DTYPES [TENSOR_DTYPES ...], --tensor-datatypes TENSOR_DTYPES [TENSOR_DTYPES ...]
                               # 每个网络 I/O 张量使用的数据类型
                               # 格式: --tensor-datatypes <tensor_name>:<tensor_datatype>
--trt-network-func-name TRT_NETWORK_FUNC_NAME
                               # [已弃用] 加载网络的函数名称，默认为 load_network
--trt-network-postprocess-script TRT_NETWORK_POSTPROCESS_SCRIPT [TRT_NETWORK_POSTPROCESS_SCRIPT ...], --trt-npps TRT_NETWORK_POSTPROCESS_SCRIPT [TRT_NETWORK_POSTPROCESS_SCRIPT ...]
                               # [实验性] 指定在解析的 TensorRT 网络上运行的后处理脚本
--strongly-typed               # 将网络标记为强类型
--mark-debug MARK_DEBUG [MARK_DEBUG ...]
                               # 指定要标记为调试张量的张量名称列表
--mark-unfused-tensors-as-debug-tensors
                               # 将未融合的张量标记为调试张量
```

### TensorRT 引擎参数
```bash
--save-timing-cache SAVE_TIMING_CACHE
                               # 构建引擎时保存策略时序缓存的路径
--load-runtime LOAD_RUNTIME    # 加载运行时的路径 (用于版本兼容引擎)
```

### ONNX-TRT 解析器标志
```bash
--onnx-flags ONNX_FLAGS [ONNX_FLAGS ...]
                               # 调整 ONNX 解析器默认解析行为的标志
--plugin-instancenorm          # 清除 trt.OnnxParserFlag.NATIVE_INSTANCENORM 标志
                               # 强制使用 ONNX InstanceNorm 的插件实现
```

### 基本用法示例
```bash
# 显示基本模型信息
polygraphy inspect model model.onnx

# 显示层信息和权重
polygraphy inspect model model.onnx --show layers weights

# 显示完整信息
polygraphy inspect model model.onnx --show layers attrs weights --list-unbounded-dds

# TensorRT 引擎分析
polygraphy inspect model model.engine --show layers weights

# 转换后分析
polygraphy inspect model model.onnx --convert-to trt --show layers
```

## 📊 inspect data - 数据文件检查

显示从 Polygraphy 的 Comparator.run() 保存的推理输入和输出信息 (例如，通过 `--save-outputs` 或 `--save-inputs` 从 `polygraphy run` 保存的输出)。

### 基本语法
```bash
polygraphy inspect data [-h] [日志参数] [-a] [-s] [--histogram] [-n NUM_ITEMS] [--line-width LINE_WIDTH] path
```

### 位置参数
```bash
path                           # 包含来自 Polygraphy 的输入或输出数据的文件路径
```

### 显示控制参数
```bash
-a, --all                      # 显示数据中所有迭代的信息，而不仅是第一个
-s, --show-values              # 显示张量的值而不仅仅是元数据
--histogram                    # 显示值分布的直方图
-n NUM_ITEMS, --num-items NUM_ITEMS
                               # 打印数组时在每个维度开始和结尾显示的值数量
                               # 使用 -1 显示数组中的所有元素，默认为 3
--line-width LINE_WIDTH        # 显示数组时每行的字符数
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
```

## 📋 inspect tactics - 策略重播文件检查

以人类可读的格式显示 Polygraphy 策略重播文件的内容，例如通过 `--save-tactics` 生成的文件。

### 基本语法
```bash
polygraphy inspect tactics [-h] [日志参数] tactic_replay
```

### 位置参数
```bash
tactic_replay                  # 策略重播文件的路径
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

## ⚙️ inspect capability - TensorRT 兼容性检查

确定 TensorRT 运行 ONNX 图的能力。图将被分区为支持和不支持的子图，或仅根据静态检查错误进行分析。

### 基本语法
```bash
polygraphy inspect capability [-h] [日志参数] [ONNX参数] [保存参数] [--with-partitioning] model_file
```

### 位置参数
```bash
model_file                     # 模型文件路径
```

### 检查选项
```bash
--with-partitioning            # 是否在解析失败的节点上对模型图进行分区
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
                               # 保存 ONNX 模型的目录路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                               # 是否将权重数据保存在外部文件中
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                               # 大小阈值 (字节)，超过此阈值的张量数据将存储在外部文件中
                               # 支持 K、M、G 后缀表示 KiB、MiB、GiB，默认 1024 字节
--no-save-all-tensors-to-one-file
                               # 保存外部数据时不将所有张量保存到单个文件
```

### 基本用法示例
```bash
# 检查 ONNX 模型的 TensorRT 兼容性
polygraphy inspect capability model.onnx

# 启用图分区分析
polygraphy inspect capability model.onnx --with-partitioning

# 详细兼容性报告
polygraphy inspect capability model.onnx --with-partitioning --verbose

# 保存支持的子图
polygraphy inspect capability model.onnx --with-partitioning -o supported_subgraphs/
```

## 🔍 inspect diff-tactics - 策略差异分析

根据好坏 Polygraphy 策略重播文件集合，确定潜在的坏 TensorRT 策略，例如通过 `--save-tactics` 保存的文件。

### 基本语法
```bash
polygraphy inspect diff-tactics [-h] [日志参数] [--dir DIR] [--good GOOD] [--bad BAD]
```

### 策略文件参数
```bash
--dir DIR                      # 包含好坏 Polygraphy 策略重播文件的目录
                               # 默认搜索名为 'good' 和 'bad' 的子目录
--good GOOD                    # 包含好策略重播文件的目录或单个好文件
--bad BAD                      # 包含坏策略重播文件的目录或单个坏文件
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
polygraphy inspect sparsity [-h] [日志参数] [ONNX加载参数] model_file
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

## 📚 相关文档

- [run - 跨框架比较](./run.md) - 使用分析结果优化运行参数
- [convert - 模型转换](./convert.md) - 基于分析结果调整转换策略
- [surgeon - 模型修改](./surgeon.md) - 根据分析结果修改模型结构

---

*`polygraphy inspect` 是理解和调试模型的第一步，详细的分析有助于后续的优化和部署决策。*