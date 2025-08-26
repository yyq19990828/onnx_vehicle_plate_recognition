# polygraphy run - 跨框架推理比较

运行推理并跨后端比较结果。

`run` 的典型使用方法是：

    polygraphy run [model_file] [runners...] [runner_options...]

`run` 将在指定的模型上使用所有指定的运行器运行推理，并比较它们之间的推理输出。

**提示**: 您可以使用 `--gen-script` 生成一个 Python 脚本，该脚本执行与 `run` 命令完全相同的操作。

## 📋 基本语法

```bash
polygraphy run [-h] [-v] [-q] [--verbosity VERBOSITY [VERBOSITY ...]]
               [--silent] [--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]]
               [--log-file LOG_FILE] [--tf] [--onnxrt] [--pluginref] [--trt]
               [--model-type {frozen,keras,ckpt,onnx,engine,uff,trt-network-script,caffe}]
               [--input-shapes INPUT_SHAPES [INPUT_SHAPES ...]]
               [多个选项...] [model_file]
```

## ⚙️ 选项参数

### 基本选项 (options)
```bash
-h, --help            # 显示此帮助消息并退出
--gen GEN_SCRIPT, --gen-script GEN_SCRIPT
                      # 保存生成的 Python 脚本的路径，该脚本将完全按照 `run` 的操作进行。
                      # 当启用此选项时，`run` 将保存脚本并退出。
                      # 使用值 `-` 将脚本打印到标准输出而不是保存到文件。
```

### 日志选项 (Logging)
```bash
-v, --verbose         # 增加日志详细程度。多次指定可获得更高的详细程度
-q, --quiet           # 减少日志详细程度。多次指定可获得更低的详细程度
--verbosity VERBOSITY [VERBOSITY ...]
                      # 要使用的日志详细程度。优先于 `-v` 和 `-q` 选项，
                      # 与它们不同的是，允许控制每个路径的详细程度。
                      # 详细程度值应来自 Polygraphy 的日志详细程度，在 `Logger` 类中定义并且不区分大小写。
                      # 例如：`--verbosity INFO` 或 `--verbosity verbose`。
                      # 要指定每个路径的详细程度，请使用格式：`<path>:<verbosity>`。
                      # 例如：`--verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE`。
                      # 路径应相对于 `polygraphy/` 目录。例如，`polygraphy/backend` 应仅指定为 `backend`。
                      # 使用最匹配的路径来确定详细程度。
--silent              # 禁用所有输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
                      # 日志消息的格式：
                      # {'timestamp': 包含时间戳, 'line-info': 包含文件和行号, 'no-colors': 禁用颜色}
--log-file LOG_FILE   # Polygraphy 日志输出应写入的文件路径。
                      # 这可能不包括来自依赖项（如 TensorRT 或 ONNX-Runtime）的日志输出。
```

### 运行器选项 (Runners)
```bash
--tf                  # 使用 TensorFlow 运行推理
--onnxrt              # 使用 ONNX-Runtime 运行推理
--pluginref           # 使用 Plugin CPU Reference 运行推理
--trt                 # 使用 TensorRT 运行推理
```

### 模型选项 (Model)
```bash
model_file            # 模型的路径
--model-type {frozen,keras,ckpt,onnx,engine,uff,trt-network-script,caffe}
                      # 输入模型的类型：
                      # {'frozen': TensorFlow frozen graph; 
                      #  'keras': Keras model; 
                      #  'ckpt': TensorFlow checkpoint directory; 
                      #  'onnx': ONNX model; 
                      #  'engine': TensorRT engine; 
                      #  'trt-network-script': 定义 `load_network` 函数的 Python 脚本，
                      #    该函数不接受参数并返回 TensorRT Builder、Network 和可选的 Parser。
                      #    如果函数名称不是 `load_network`，可以在模型文件后用冒号分隔指定。
                      #    例如：`my_custom_script.py:my_func`; 
                      #  'uff': UFF file [deprecated]; 
                      #  'caffe': Caffe prototxt [deprecated]}
--input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --inputs INPUT_SHAPES [INPUT_SHAPES ...]
                      # 模型输入及其形状。用于确定生成推理输入数据时要使用的形状。
                      # 格式：--input-shapes <name>:<shape>
                      # 例如：--input-shapes image:[1,3,224,224] other_input:[10]
```

### TensorFlow-TensorRT 集成 ([UNTESTED] TensorFlow-TensorRT Integration)
```bash
--tftrt, --use-tftrt  # 启用 TF-TRT 集成
--minimum-segment-size MINIMUM_SEGMENT_SIZE
                      # 转换为 TensorRT 的段的最小长度
--dynamic-op          # 启用动态模式（将引擎构建延迟到运行时）
```

### TensorFlow 模型加载 (TensorFlow Model Loading)
```bash
--ckpt CKPT           # [实验性] 要加载的检查点名称。
                      # 如果缺少 `checkpoint` 文件，则为必需。
                      # 不应包含文件扩展名（例如，要加载 `model.meta`，使用 `--ckpt=model`）
--tf-outputs TF_OUTPUTS [TF_OUTPUTS ...]
                      # TensorFlow 输出的名称。使用 `--tf-outputs mark all` 
                      # 表示所有张量都应用作输出
--save-pb SAVE_FROZEN_GRAPH_PATH
                      # 保存 TensorFlow 冻结 graphdef 的路径
--save-tensorboard SAVE_TENSORBOARD_PATH
                      # [实验性] 保存 TensorBoard 可视化的路径
--freeze-graph        # [实验性] 尝试冻结图
```

### TensorFlow 会话配置 (TensorFlow Session Configuration)
```bash
--gpu-memory-fraction GPU_MEMORY_FRACTION
                      # TensorFlow 每个进程可分配的 GPU 内存的最大百分比
--allow-growth        # 允许 TensorFlow 分配的 GPU 内存增长
--xla                 # [实验性] 尝试使用 XLA 运行图
```

### TensorFlow 推理 (TensorFlow Inference)
```bash
--save-timeline SAVE_TIMELINE
                      # [实验性] 保存用于性能分析推理的时间轴 JSON 文件的目录（在 chrome://tracing 查看）
```

### TensorFlow-ONNX 模型转换 (TensorFlow-ONNX Model Conversion)
```bash
--opset OPSET         # 转换为 ONNX 时要使用的 Opset
```

### ONNX 模型保存 (ONNX Model Saving)
```bash
--save-onnx SAVE_ONNX # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                      # 是否将权重数据保存在外部文件中。
                      # 要使用非默认路径，请将所需路径作为参数提供。
                      # 这始终是相对路径；外部数据始终写入与模型相同的目录。
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                      # 大小阈值（以字节为单位），超过此阈值的张量数据将存储在外部文件中。
                      # 小于此阈值的张量将保留在 ONNX 文件中。
                      # 可选地，使用 `K`、`M` 或 `G` 后缀表示 KiB、MiB 或 GiB。
                      # 例如，`--external-data-size-threshold=16M` 等价于 
                      # `--external-data-size-threshold=16777216`。
                      # 如果未设置 `--save-external-data`，则无效果。默认值为 1024 字节。
--no-save-all-tensors-to-one-file
                      # 保存外部数据时不要将所有张量保存到单个文件。
                      # 如果未设置 `--save-external-data`，则无效果
```

### ONNX 形状推理 (ONNX Shape Inference)
```bash
--shape-inference, --do-shape-inference
                      # 加载模型时启用 ONNX 形状推理
--no-onnxruntime-shape-inference
                      # 禁用使用 ONNX-Runtime 的形状推理实用程序。
                      # 这将强制 Polygraphy 使用 `onnx.shape_inference`。
                      # 注意，ONNX-Runtime 的形状推理实用程序可能更高效且内存友好。
```

### ONNX 模型加载 (ONNX Model Loading)
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                      # 包含模型外部数据的目录路径。
                      # 通常，只有当外部数据未存储在模型目录中时才需要。
--ignore-external-data
                      # 忽略外部数据，仅加载没有任何权重的模型结构。
                      # 模型仅可用于不需要权重的目的，例如提取子图或检查模型结构。
                      # 这在外部数据不可用的情况下很有用。
--onnx-outputs ONNX_OUTPUTS [ONNX_OUTPUTS ...]
                      # 要标记为输出的 ONNX 张量名称。使用特殊值 'mark all' 表示所有张量都应用作输出
--onnx-exclude-outputs ONNX_EXCLUDE_OUTPUTS [ONNX_EXCLUDE_OUTPUTS ...]
                      # [实验性] 要取消标记为输出的 ONNX 输出名称
--fp-to-fp16          # 将 ONNX 模型中的所有浮点张量转换为 16 位精度。
                      # 这 *不是* 使用 TensorRT 的 fp16 精度所必需的，但对其他后端可能有用。
                      # 需要 onnxmltools。
```

### ONNX-Runtime 会话创建 (ONNX-Runtime Session Creation)
```bash
--providers PROVIDERS [PROVIDERS ...], --execution-providers PROVIDERS [PROVIDERS ...]
                      # 要按优先级顺序使用的执行提供程序列表。
                      # 每个提供程序可以是 ONNX-Runtime 中可用执行提供程序的精确匹配
                      # 或不区分大小写的部分匹配。
                      # 例如，值 'cpu' 将匹配 'CPUExecutionProvider'
```

### TensorRT 构建器配置 (TensorRT Builder Configuration)
```bash
--trt-min-shapes TRT_MIN_SHAPES [TRT_MIN_SHAPES ...]
                      # 优化配置文件将支持的最小形状。为每个配置文件指定一次此选项。
                      # 如果未提供，则使用推理时输入形状。
                      # 格式：--trt-min-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]
--trt-opt-shapes TRT_OPT_SHAPES [TRT_OPT_SHAPES ...]
                      # 优化配置文件最佳性能的形状。为每个配置文件指定一次此选项。
                      # 如果未提供，则使用推理时输入形状。
                      # 格式：--trt-opt-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]
--trt-max-shapes TRT_MAX_SHAPES [TRT_MAX_SHAPES ...]
                      # 优化配置文件将支持的最大形状。为每个配置文件指定一次此选项。
                      # 如果未提供，则使用推理时输入形状。
                      # 格式：--trt-max-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]
--tf32                # 在 TensorRT 中启用 tf32 精度
--fp16                # 在 TensorRT 中启用 fp16 精度
--bf16                # 在 TensorRT 中启用 bf16 精度
--fp8                 # 在 TensorRT 中启用 fp8 精度
--int8                # 在 TensorRT 中启用 int8 精度。如果需要校准但未提供校准缓存，
                      # 此选项将使 TensorRT 使用 Polygraphy 数据加载器提供校准数据来运行 int8 校准。
                      # 如果运行校准且模型具有动态形状，则将使用最后一个优化配置文件作为校准配置文件。
--precision-constraints {prefer,obey,none}
                      # 如果设置为 `prefer`，TensorRT 将限制可用策略为网络中指定的层精度，
                      # 除非不存在具有首选层约束的实现，在这种情况下它将发出警告并使用最快的可用实现。
                      # 如果设置为 `obey`，TensorRT 将在不存在具有首选层约束的实现时构建网络失败。
                      # 默认为 `none`
--sparse-weights      # 在 TensorRT 中启用稀疏权重优化
--version-compatible  # 构建一个设计为向前 TensorRT 版本兼容的引擎
--exclude-lean-runtime
                      # 在启用版本兼容性时从计划中排除精简运行时
--calibration-cache CALIBRATION_CACHE
                      # 加载/保存校准缓存的路径。用于存储校准比例以加速 int8 校准过程。
                      # 如果提供的路径尚不存在，将在引擎构建期间计算并写入 int8 校准比例。
                      # 如果提供的路径存在，将读取它并在引擎构建期间跳过 int8 校准。
--calib-base-cls CALIBRATION_BASE_CLASS, --calibration-base-class CALIBRATION_BASE_CLASS
                      # 要使用的校准基类名称。例如，'IInt8MinMaxCalibrator'。
--quantile QUANTILE   # 用于 IInt8LegacyCalibrator 的分位数。对其他校准器类型无效果。
--regression-cutoff REGRESSION_CUTOFF
                      # 用于 IInt8LegacyCalibrator 的回归截止。对其他校准器类型无效果。
--load-timing-cache LOAD_TIMING_CACHE
                      # 加载策略时序缓存的路径。用于缓存策略时序信息以加速引擎构建过程。
                      # 如果 --load-timing-cache 指定的文件不存在，Polygraphy 将发出警告并
                      # 回退到使用空时序缓存。
--error-on-timing-cache-miss
                      # 当正在计时的策略不存在于时序缓存中时发出错误
--disable-compilation-cache
                      # 禁用缓存 JIT 编译的代码
--save-tactics SAVE_TACTICS, --save-tactic-replay SAVE_TACTICS
                      # 保存 Polygraphy 策略重播文件的路径。
                      # 将记录有关 TensorRT 选择的策略的详细信息并作为 JSON 文件存储在此位置。
--load-tactics LOAD_TACTICS, --load-tactic-replay LOAD_TACTICS
                      # 加载 Polygraphy 策略重播文件的路径，例如由 --save-tactics 创建的文件。
                      # 文件中指定的策略将用于覆盖 TensorRT 的默认选择。
--tactic-sources [TACTIC_SOURCES ...]
                      # 要启用的策略源。这控制 TensorRT 允许从哪些库（例如 cudnn、cublas 等）
                      # 加载策略。值来自 trt.TacticSource 枚举中值的名称，不区分大小写。
                      # 如果未提供参数，例如 '--tactic-sources'，则禁用所有策略源。
                      # 默认为 TensorRT 的默认策略源。
--trt-config-script TRT_CONFIG_SCRIPT
                      # 定义创建 TensorRT IBuilderConfig 的函数的 Python 脚本的路径。
                      # 该函数应接受 builder 和 network 作为参数并返回 TensorRT builder 配置。
                      # 指定此选项时，将忽略所有其他配置参数。默认情况下，Polygraphy 查找名为 `load_config` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。例如：`my_custom_script.py:my_func`
--trt-config-func-name TRT_CONFIG_FUNC_NAME
                      # [已弃用 - 函数名称可以用 --trt-config-script 指定，如：`my_custom_script.py:my_func`] 
                      # 使用 trt-config-script 时，这指定创建配置的函数的名称。默认为 `load_config`。
--trt-config-postprocess-script TRT_CONFIG_POSTPROCESS_SCRIPT, --trt-cpps TRT_CONFIG_POSTPROCESS_SCRIPT
                      # [实验性] 定义修改 TensorRT IBuilderConfig 的函数的 Python 脚本的路径。
                      # 此函数将在 Polygraphy 完成创建构建器配置后调用，应接受 builder、network 和 config 作为参数
                      # 并就地修改配置。与 `--trt-config-script` 不同，所有其他配置参数将反映在传递给函数的配置中。
                      # 默认情况下，Polygraphy 查找名为 `postprocess_config` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。例如：`my_custom_script.py:my_func`
--trt-safety-restricted
                      # 在 TensorRT 中启用安全范围检查
--refittable          # 启用引擎在构建后可以用新权重重新拟合
--strip-plan          # 构建时剥离可重新拟合权重的引擎
--use-dla             # [实验性] 使用 DLA 作为默认设备类型
--allow-gpu-fallback  # [实验性] 允许 DLA 上不支持的层回退到 GPU。如果未设置 --use-dla，则无效果。
--pool-limit MEMORY_POOL_LIMIT [MEMORY_POOL_LIMIT ...], --memory-pool-limit MEMORY_POOL_LIMIT [MEMORY_POOL_LIMIT ...]
                      # 内存池限制。内存池名称来自 `trt.MemoryPoolType` 枚举中值的名称，不区分大小写。
                      # 格式：`--pool-limit <pool_name>:<pool_limit> ...`。
                      # 例如，`--pool-limit dla_local_dram:1e9 workspace:16777216`。
                      # 可选地，使用 `K`、`M` 或 `G` 后缀表示 KiB、MiB 或 GiB。
                      # 例如，`--pool-limit workspace:16M` 等价于 `--pool-limit workspace:16777216`。
--preview-features [PREVIEW_FEATURES ...]
                      # 要启用的预览功能。值来自 trt.PreviewFeature 枚举中值的名称，不区分大小写。
                      # 如果未提供参数，例如 '--preview-features'，则禁用所有预览功能。
                      # 默认为 TensorRT 的默认预览功能。
--builder-optimization-level BUILDER_OPTIMIZATION_LEVEL
                      # 构建器优化级别。设置更高的优化级别允许优化器花费更多时间寻找优化机会。
                      # 与较低优化级别构建的引擎相比，生成的引擎可能具有更好的性能。
                      # 请参考 TensorRT API 文档了解详细信息。
--hardware-compatibility-level HARDWARE_COMPATIBILITY_LEVEL
                      # 用于引擎的硬件兼容级别。这允许在一种 GPU 架构上构建的引擎在其他架构的 GPU 上工作。
                      # 值来自 `trt.HardwareCompatibilityLevel` 枚举中值的名称，不区分大小写。
                      # 例如，`--hardware-compatibility-level ampere_plus`
--max-aux-streams MAX_AUX_STREAMS
                      # 允许 TensorRT 使用的最大辅助流数。如果网络包含可以并行运行的操作符，
                      # TRT 可以除了提供给 IExecutionContext.execute_async_v3() 调用的流之外，使用辅助流执行它们。
                      # 辅助流的默认最大数量由 TensorRT 中的启发式确定，基于启用多流是否会提高性能。
                      # 请参考 TensorRT API 文档了解详细信息。
--quantization-flags [QUANTIZATION_FLAGS ...]
                      # 要启用的 Int8 量化标志。值来自 trt.QuantizationFlag 枚举中值的名称，不区分大小写。
                      # 如果未提供参数，例如 '--quantization-flags'，则禁用所有量化标志。
                      # 默认为 TensorRT 的默认量化标志。
--profiling-verbosity PROFILING_VERBOSITY
                      # 生成的引擎中 NVTX 注释的详细程度。值来自 `trt.ProfilingVerbosity` 枚举中值的名称，不区分大小写。
                      # 例如，`--profiling-verbosity detailed`。默认为 'detailed'。
--weight-streaming    # 构建权重可流式传输的引擎。必须与 --strongly-typed 一起设置。
                      # 权重流式传输量可以用 --weight-streaming-budget 设置。
--runtime-platform RUNTIME_PLATFORM
                      # TensorRT 引擎执行的目标运行时平台（操作系统和 CPU 架构）。
                      # 当目标运行时平台与构建平台不同时，TensorRT 提供跨平台引擎兼容性支持。
                      # 值来自 `trt.RuntimePlatform` 枚举中值的名称，不区分大小写。
                      # 例如，`--runtime-platform same_as_build`，`--runtime-platform windows_amd64`
--tiling-optimization-level TILING_OPTIMIZATION_LEVEL
                      # 平铺优化级别。设置更高的优化级别允许 TensorRT 花费更多构建时间进行更多平铺策略。
                      # 值来自 `trt.TilingOptimizationLevel` 枚举中值的名称，不区分大小写。
```

### TensorRT 插件加载 (TensorRT Plugin Loading)
```bash
--plugins PLUGINS [PLUGINS ...]
                      # 要加载的插件库路径
```

### ONNX-TRT 解析器标志 (ONNX-TRT Parser Flags)
```bash
--onnx-flags ONNX_FLAGS [ONNX_FLAGS ...]
                      # 用于调整 ONNX 解析器默认解析行为的标志。标志值来自 `trt.OnnxParserFlag` 枚举，不区分大小写。
                      # 例如：`--onnx-flags native_instancenorm`
--plugin-instancenorm # 切换清除 `trt.OnnxParserFlag.NATIVE_INSTANCENORM` 标志并强制使用 ONNX InstanceNorm 的插件实现。
                      # 注意，从 TensorRT 10.0 开始，默认启用 `trt.OnnxParserFlag.NATIVE_INSTANCENORM`。
```

### TensorRT 网络加载 (TensorRT Network Loading)
```bash
--trt-outputs TRT_OUTPUTS [TRT_OUTPUTS ...]
                      # TensorRT 输出的名称。使用 '--trt-outputs mark all' 表示所有张量都应用作输出
--trt-exclude-outputs TRT_EXCLUDE_OUTPUTS [TRT_EXCLUDE_OUTPUTS ...]
                      # [实验性] 要取消标记为输出的 TensorRT 输出名称
--layer-precisions LAYER_PRECISIONS [LAYER_PRECISIONS ...]
                      # 每层要使用的计算精度。应按每层指定，使用格式：`--layer-precisions <layer_name>:<layer_precision>`。
                      # 精度值来自 TensorRT 数据类型别名，如 float32、float16、int8、bool 等。
                      # 例如：`--layer-precisions example_layer:float16 other_layer:int8`。
                      # 提供此选项时，还应将 --precision-constraints 设置为 'prefer' 或 'obey'。
--tensor-dtypes TENSOR_DTYPES [TENSOR_DTYPES ...], --tensor-datatypes TENSOR_DTYPES [TENSOR_DTYPES ...]
                      # 每个网络 I/O 张量要使用的数据类型。应按每个张量指定，
                      # 使用格式：`--tensor-datatypes <tensor_name>:<tensor_datatype>`。
                      # 数据类型值来自 TensorRT 数据类型别名，如 float32、float16、int8、bool 等。
                      # 例如：`--tensor-datatypes example_tensor:float16 other_tensor:int8`。
--trt-network-func-name TRT_NETWORK_FUNC_NAME
                      # [已弃用 - 函数名称可以与脚本一起指定，如：`my_custom_script.py:my_func`] 
                      # 当使用 trt-network-script 而不是其他模型类型时，这指定加载网络的函数名称。默认为 `load_network`。
--trt-network-postprocess-script TRT_NETWORK_POSTPROCESS_SCRIPT [TRT_NETWORK_POSTPROCESS_SCRIPT ...], --trt-npps TRT_NETWORK_POSTPROCESS_SCRIPT [TRT_NETWORK_POSTPROCESS_SCRIPT ...]
                      # [实验性] 指定要在解析的 TensorRT 网络上运行的后处理脚本。
                      # 脚本文件可以选择性地与要调用的可调用名称一起后缀。
                      # 例如：`--trt-npps process.py:do_something`。
                      # 如果未指定可调用，则默认情况下 Polygraphy 使用可调用名称 `postprocess`。
                      # 可调用预期接受一个名为 `network` 的命名参数，类型为 `trt.INetworkDefinition`。
                      # 可以指定多个脚本，在这种情况下按给定顺序执行。
--strongly-typed      # 将网络标记为强类型
--mark-debug MARK_DEBUG [MARK_DEBUG ...]
                      # 指定要标记为调试张量的张量名称列表。例如，`--mark-debug tensor1 tensor2 tensor3`。
--mark-unfused-tensors-as-debug-tensors
                      # 将未融合的张量标记为调试张量
```

### TensorRT 引擎保存 (TensorRT Engine Saving)
```bash
--save-engine SAVE_ENGINE
                      # 保存 TensorRT 引擎的路径
```

### TensorRT 引擎 (TensorRT Engine)
```bash
--save-timing-cache SAVE_TIMING_CACHE
                      # 如果构建引擎，保存策略时序缓存的路径。现有缓存将与任何新收集的时序信息一起追加。
--load-runtime LOAD_RUNTIME
                      # 从中加载运行时的路径，该运行时可用于加载排除精简运行时的版本兼容引擎
```

### TensorRT 推理 (TensorRT Inference)
```bash
--optimization-profile OPTIMIZATION_PROFILE
                      # 用于推理的优化配置文件的索引
--allocation-strategy {static,profile,runtime}
                      # 激活内存分配方式。static：基于所有配置文件的最大可能大小进行预分配。
                      # profile：分配配置文件使用所需的内容。runtime：分配当前输入形状所需的内容。
--weight-streaming-budget WEIGHT_STREAMING_BUDGET
                      # TensorRT 在运行时可用于权重的 GPU 内存字节数。引擎必须在启用权重流的情况下构建。
                      # 它可以采用以下值：None 或 -2：在运行时禁用权重流。-1：TensorRT 将自动决定流预算。
                      # 0 到 100%：TRT 保留在 GPU 上的权重百分比。0% 将流式传输最大数量的权重。
                      # >=0B：驻留在 GPU 上的可流式传输权重的确切数量（支持单位后缀）。
```

### 数据加载器 (Data Loader)
```bash
--seed SEED           # 用于随机输入的种子
--val-range VAL_RANGE [VAL_RANGE ...]
                      # 数据加载器中生成的值范围。要指定每个输入的范围，
                      # 使用格式：`--val-range <input_name>:[min,max]`。
                      # 如果未提供输入名称，则范围用于任何未明确指定的输入。
                      # 例如：`--val-range [0,1] inp0:[2,50] inp1:[3.0,4.6]`
--int-min INT_MIN     # [已弃用：使用 --val-range] 随机整数输入的最小整数值
--int-max INT_MAX     # [已弃用：使用 --val-range] 随机整数输入的最大整数值
--float-min FLOAT_MIN # [已弃用：使用 --val-range] 随机浮点输入的最小浮点值
--float-max FLOAT_MAX # [已弃用：使用 --val-range] 随机浮点输入的最大浮点值
--iterations NUM, --iters NUM
                      # 默认数据加载器应提供数据的推理迭代次数
--data-loader-backend-module {numpy,torch}
                      # 用于生成输入数组的模块。当前支持的选项：numpy, torch
--load-inputs LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...], --load-input-data LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...]
                      # 加载输入的路径。文件应该是 JSON 化的 List[Dict[str, numpy.ndarray]]，
                      # 即一个列表，其中每个元素是单次迭代的 feed_dict。
                      # 使用此选项时，将忽略所有其他数据加载器参数。
--data-loader-script DATA_LOADER_SCRIPT
                      # 定义加载输入数据函数的 Python 脚本的路径。
                      # 函数应不接受参数并返回产生输入数据（Dict[str, np.ndarray]）的生成器或可迭代对象。
                      # 使用此选项时，将忽略所有其他数据加载器参数。
                      # 默认情况下，Polygraphy 查找名为 `load_data` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。例如：`my_custom_script.py:my_func`
--data-loader-func-name DATA_LOADER_FUNC_NAME
                      # [已弃用 - 函数名称可以用 --data-loader-script 指定，如：`my_custom_script.py:my_func`] 
                      # 使用 data-loader-script 时，这指定加载数据的函数名称。默认为 `load_data`。
```

### 比较器推理 (Comparator Inference)
```bash
--warm-up NUM         # 计时推理之前的预热运行次数
--use-subprocess      # 在隔离的子进程中运行运行器。不能与调试器一起使用
--save-inputs SAVE_INPUTS_PATH, --save-input-data SAVE_INPUTS_PATH
                      # 保存推理输入的路径。输入（List[Dict[str, numpy.ndarray]]）将编码为 JSON 并保存
--save-outputs SAVE_OUTPUTS_PATH, --save-results SAVE_OUTPUTS_PATH
                      # 保存运行器结果的路径。结果（RunResults）将编码为 JSON 并保存
```

### 比较器后处理 (Comparator Postprocessing)
```bash
--postprocess POSTPROCESS [POSTPROCESS ...], --postprocess-func POSTPROCESS [POSTPROCESS ...]
                      # 在比较之前对指定输出应用后处理。格式：`--postprocess [<out_name>:]<func>`。
                      # 如果未提供输出名称，该函数将应用于所有输出。
                      # 例如：`--postprocess out0:top-5 out1:top-3` 或 `--postprocess top-5`。
                      # 可用的后处理函数：`top-<K>[,axis=<axis>]`：沿指定轴（默认为最后一个轴）
                      # 取 K 个最高值的索引，其中 K 是整数。例如：`--postprocess top-5` 或 `--postprocess top-5,axis=1`
```

### 比较器比较 (Comparator Comparisons)
```bash
--validate            # 检查输出中的 NaN 和 Inf
--fail-fast           # 快速失败（第一次失败后停止比较）
--compare {simple,indices}, --compare-func {simple,indices}
                      # 用于执行比较的函数名称。有关详细信息，请参见 `CompareFunc` 的 API 文档。默认为 'simple'。
--compare-func-script COMPARE_FUNC_SCRIPT
                      # [实验性] 定义可以比较两个迭代结果的函数的 Python 脚本的路径。
                      # 此函数必须具有签名：`(IterationResult, IterationResult) -> OrderedDict[str, bool]`。
                      # 有关详细信息，请参见 `Comparator.compare_accuracy()` 的 API 文档。
                      # 如果提供，这将覆盖所有其他比较函数选项。
                      # 默认情况下，Polygraphy 查找名为 `compare_outputs` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。例如：`my_custom_script.py:my_func`
--load-outputs LOAD_OUTPUTS_PATHS [LOAD_OUTPUTS_PATHS ...], --load-results LOAD_OUTPUTS_PATHS [LOAD_OUTPUTS_PATHS ...]
                      # 在比较之前从运行器加载结果的路径。每个文件应该是 JSON 化的 RunResults
```

### 比较函数: simple (Comparison Function: `simple`)
```bash
--no-shape-check      # 禁用检查输出形状完全匹配
--rtol RTOL [RTOL ...], --rel-tol RTOL [RTOL ...]
                      # 输出比较的相对容差。这表示为第二组输出值的百分比。
                      # 例如，值 0.01 将检查第一组输出是否在第二组的 1% 以内。
                      # 要指定每个输出的容差，使用格式：`--rtol [<out_name>:]<rtol>`。
                      # 如果未提供输出名称，则容差用于任何未明确指定的输出。
                      # 例如：`--rtol 1e-5 out0:1e-4 out1:1e-3`。
                      # 注意，默认容差通常适用于 FP32，但对于 FP16 或 INT8 等较低精度可能太严格。
--atol ATOL [ATOL ...], --abs-tol ATOL [ATOL ...]
                      # 输出比较的绝对容差。要指定每个输出的容差，使用格式：`--atol [<out_name>:]<atol>`。
                      # 如果未提供输出名称，则容差用于任何未明确指定的输出。
                      # 例如：`--atol 1e-5 out0:1e-4 out1:1e-3`。
                      # 注意，默认容差通常适用于 FP32，但对于 FP16 或 INT8 等较低精度可能太严格。
--check-error-stat CHECK_ERROR_STAT [CHECK_ERROR_STAT ...]
                      # 要检查的误差统计。有关可能值的详细信息，请参见 CompareFunc.simple() 的文档。
                      # 要指定每个输出的值，使用格式：`--check-error-stat [<out_name>:]<stat>`。
                      # 如果未提供输出名称，则值用于任何未明确指定的输出。
                      # 例如：`--check-error-stat max out0:mean out1:median`
--infinities-compare-equal
                      # 如果设置，则输出中的任何匹配 ±inf 值将具有 0 的 absdiff。
                      # 否则，默认情况下它们将具有 NaN 的 absdiff。
--save-heatmaps SAVE_HEATMAPS
                      # [实验性] 保存绝对和相对误差热图的目录
--show-heatmaps       # [实验性] 是否显示绝对和相对误差的热图。默认为 False。
--save-error-metrics-plot SAVE_ERROR_METRICS_PLOT
                      # [实验性] 保存误差指标图的目录路径。如果设置，生成绝对和相对误差与参考输出幅度的图。
                      # 如果目录不存在，则创建该目录。这对于查找误差趋势、确定精度失败是否只是异常值或更深层次的问题很有用。
--show-error-metrics-plot
                      # [实验性] 是否显示误差指标图。默认为 False。
--error-quantile ERROR_QUANTILE [ERROR_QUANTILE ...]
                      # 要比较的误差分位数。浮点数，有效范围 [0, 1]。
                      # 要指定每个输出的值，使用格式：`--quantile [<out_name>:]<stat>`。
                      # 如果未提供输出名称，则值用于任何未明确指定的输出。
                      # 例如：`--error-quantile 0.95 out0:0.8 out1:0.9`
```

### 比较函数: indices (Comparison Function: `indices`)
```bash
--index-tolerance INDEX_TOLERANCE [INDEX_TOLERANCE ...]
                      # 输出比较的索引容差。有关其含义的详细信息，请参见 `CompareFunc.indices()` 的 API 文档。
                      # 要指定每个输出的容差，使用格式：`--index-tolerance [<out_name>:]<index_tol>`。
                      # 如果未提供输出名称，则容差用于任何未明确指定的输出。
                      # 例如：`--index_tolerance 1 out0:0 out1:3`。
```

## 💡 使用示例

### 1. 基础精度验证
```bash
# 简单的 ONNX Runtime vs TensorRT 比较
polygraphy run resnet50.onnx --onnxrt --trt

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

# 使用自定义数据加载脚本
polygraphy run model.onnx --onnxrt --trt \
  --data-loader-script custom_loader.py:my_load_data
```

### 5. 逐层精度分析
```bash
# 标记所有层输出
polygraphy run model.onnx --onnxrt --trt --onnx-outputs "mark all" \
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

### 7. 生成调试脚本
```bash
# 生成等效的 Python 脚本
polygraphy run model.onnx --onnxrt --trt \
  --gen-script debug_comparison.py

# 打印脚本到标准输出
polygraphy run model.onnx --onnxrt --trt --gen-script -
```

### 8. 误差分析和可视化
```bash
# 详细误差分析
polygraphy run model.onnx --onnxrt --trt \
  --validate --fail-fast \
  --save-heatmaps error_analysis/ \
  --save-error-metrics-plot plots/ \
  --check-error-stat max mean \
  --error-quantile 0.95

# 索引比较模式
polygraphy run model.onnx --onnxrt --trt \
  --compare indices --index-tolerance 1
```

### 9. 多框架全面比较
```bash
# 包含多个推理后端
polygraphy run model.onnx --tf --onnxrt --trt \
  --warm-up 5 --iterations 20 \
  --save-outputs multi_backend_results.json
```

### 10. 性能优化测试
```bash
# 完整性能测试配置
polygraphy run model.onnx --onnxrt --trt --fp16 \
  --warm-up 10 --iterations 100 \
  --save-timing-cache timing.cache \
  --builder-optimization-level 5 \
  --tactic-sources cublas cudnn \
  --max-aux-streams 4 \
  --allocation-strategy runtime
```

## 📚 相关文档

- [convert - 模型转换](./convert.md) - 模型格式转换
- [debug - 调试工具](./debug.md) - 进一步调试失败案例  
- [inspect - 模型分析](./inspect.md) - 分析模型结构

---

*`polygraphy run` 是发现和调试推理问题的第一步，掌握其用法对于模型部署至关重要。*