# polygraphy convert - 模型格式转换

将模型转换为其他格式。

## 📋 基本语法

```bash
polygraphy convert [-h] [-v] [-q] [--verbosity VERBOSITY [VERBOSITY ...]] 
                   [--silent] [--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]]
                   [--log-file LOG_FILE] [--model-type {frozen,keras,ckpt,onnx,engine,uff,trt-network-script,caffe}]
                   [多个选项...] -o OUTPUT [--convert-to {onnx,trt,onnx-like-trt-network}] 
                   model_file
```

## ⚙️ 选项参数

### 基本选项 (options)
```bash
-h, --help            # 显示此帮助消息并退出
-o OUTPUT, --output OUTPUT
                      # 保存转换模型的路径
--convert-to {onnx,trt,onnx-like-trt-network}
                      # 尝试将模型转换为的格式。'onnx-like-trt-network' 是实验性的，
                      # 将 TensorRT 网络转换为可用于可视化的格式。
                      # 详见 'OnnxLikeFromNetwork'。
```

### 日志选项 (Logging)
```bash
-v, --verbose         # 增加日志详细程度。可多次指定以获得更高详细程度
-q, --quiet           # 降低日志详细程度。可多次指定以获得更低详细程度
--verbosity VERBOSITY [VERBOSITY ...]
                      # 要使用的日志详细程度。优先于 `-v` 和 `-q` 选项，
                      # 与它们不同，允许您控制每个路径的详细程度。
                      # 详细程度值应来自 Logger 类中定义的 Polygraphy 日志详细程度，
                      # 不区分大小写。例如：`--verbosity INFO` 或 `--verbosity verbose`。
                      # 要指定每个路径的详细程度，使用格式：`<path>:<verbosity>`。
                      # 例如：`--verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE`
                      # 路径应相对于 `polygraphy/` 目录。
                      # 例如，`polygraphy/backend` 应指定为 `backend`。
                      # 使用最接近匹配的路径来确定详细程度。
--silent              # 禁用所有输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
                      # 日志消息格式：
                      # {'timestamp': 包含时间戳, 'line-info': 包含文件和行号, 'no-colors': 禁用颜色}
--log-file LOG_FILE   # Polygraphy 日志输出应写入的文件路径。
                      # 这可能不包括来自依赖项（如 TensorRT 或 ONNX-Runtime）的日志输出。
```

### 模型选项 (Model)
```bash
model_file            # 模型路径
--model-type {frozen,keras,ckpt,onnx,engine,uff,trt-network-script,caffe}
                      # 输入模型的类型：
                      # {'frozen': TensorFlow frozen graph; 
                      #  'keras': Keras model; 
                      #  'ckpt': TensorFlow checkpoint directory; 
                      #  'onnx': ONNX model; 
                      #  'engine': TensorRT engine; 
                      #  'trt-network-script': 定义 `load_network` 函数的 Python 脚本，
                      #    该函数不接受参数并返回 TensorRT Builder、Network 和可选的 Parser。
                      #    如果函数名不是 `load_network`，可以在模型文件后用冒号分隔指定。
                      #    例如：`my_custom_script.py:my_func`; 
                      #  'uff': UFF file [deprecated]; 
                      #  'caffe': Caffe prototxt [deprecated]}
--input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --inputs INPUT_SHAPES [INPUT_SHAPES ...]
                      # 模型输入及其形状。用于确定在为推理生成输入数据时使用的形状。
                      # 格式：--input-shapes <name>:<shape>
                      # 例如：--input-shapes image:[1,3,224,224] other_input:[10]
```

### TensorFlow 模型加载 (TensorFlow Model Loading)
```bash
--ckpt CKPT           # [实验性] 要加载的检查点名称。
                      # 如果 `checkpoint` 文件缺失则为必需。
                      # 不应包含文件扩展名（例如要加载 `model.meta` 使用 `--ckpt=model`）
--tf-outputs TF_OUTPUTS [TF_OUTPUTS ...]
                      # TensorFlow 输出的名称。使用 '--tf-outputs mark all' 
                      # 表示所有张量都应用作输出
--freeze-graph        # [实验性] 尝试冻结图
```

### TensorFlow-ONNX 模型转换 (TensorFlow-ONNX Model Conversion)
```bash
--opset OPSET         # 转换为 ONNX 时使用的 Opset
```

### ONNX 形状推理 (ONNX Shape Inference)
```bash
--shape-inference, --do-shape-inference
                      # 加载模型时启用 ONNX 形状推理
--no-onnxruntime-shape-inference
                      # 禁用使用 ONNX-Runtime 的形状推理工具。
                      # 这将强制 Polygraphy 使用 `onnx.shape_inference`。
                      # 注意 ONNX-Runtime 的形状推理工具可能更高性能和内存效率。
```

### ONNX 模型加载 (ONNX Model Loading)
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                      # 包含模型外部数据的目录路径。
                      # 通常，只有在外部数据未存储在模型目录中时才需要此选项。
--ignore-external-data
                      # 忽略外部数据，仅加载模型结构而不加载任何权重。
                      # 该模型仅可用于不需要权重的目的，例如提取子图或检查模型结构。
                      # 在外部数据不可用的情况下，这可能很有用。
--onnx-outputs ONNX_OUTPUTS [ONNX_OUTPUTS ...]
                      # 要标记为输出的 ONNX 张量的名称。
                      # 使用特殊值 'mark all' 表示所有张量都应用作输出
--onnx-exclude-outputs ONNX_EXCLUDE_OUTPUTS [ONNX_EXCLUDE_OUTPUTS ...]
                      # [实验性] 要取消标记为输出的 ONNX 输出的名称。
--fp-to-fp16          # 将 ONNX 模型中的所有浮点张量转换为 16 位精度。
                      # 这不是使用 TensorRT fp16 精度所必需的，但对其他后端可能有用。
                      # 需要 onnxmltools。
```

### ONNX 模型保存 (ONNX Model Saving)
```bash
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                      # 是否将权重数据保存在外部文件中。
                      # 要使用非默认路径，提供所需路径作为参数。
                      # 这始终是相对路径；外部数据总是写入与模型相同的目录。
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                      # 大小阈值（以字节为单位），超过此阈值的张量数据将存储在外部文件中。
                      # 小于此阈值的张量将保留在 ONNX 文件中。
                      # 可选择使用 `K`、`M` 或 `G` 后缀来表示 KiB、MiB 或 GiB。
                      # 例如，`--external-data-size-threshold=16M` 等于 
                      # `--external-data-size-threshold=16777216`。
                      # 如果未设置 `--save-external-data` 则无效果。默认为 1024 字节。
--no-save-all-tensors-to-one-file
                      # 保存外部数据时不要将所有张量保存到单个文件中。
                      # 如果未设置 `--save-external-data` 则无效果
```

### 数据加载器 (Data Loader)
```bash
--seed SEED           # 用于随机输入的种子
--val-range VAL_RANGE [VAL_RANGE ...]
                      # 在数据加载器中生成的值范围。
                      # 要指定每个输入的范围，使用格式：--val-range <input_name>:[min,max]
                      # 如果未提供输入名称，范围将用于任何未明确指定的输入。
                      # 例如：--val-range [0,1] inp0:[2,50] inp1:[3.0,4.6]
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
                      # 即列表，其中每个元素都是单次迭代的 feed_dict。
                      # 使用此选项时，将忽略所有其他数据加载器参数。
--data-loader-script DATA_LOADER_SCRIPT
                      # 定义加载输入数据函数的 Python 脚本路径。
                      # 函数应不接受参数并返回生成输入数据的生成器或可迭代对象 (Dict[str, np.ndarray])。
                      # 使用此选项时，将忽略所有其他数据加载器参数。
                      # 默认情况下，Polygraphy 查找名为 `load_data` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。
                      # 例如：`my_custom_script.py:my_func`
--data-loader-func-name DATA_LOADER_FUNC_NAME
                      # [已弃用 - 可以使用 --data-loader-script 指定函数名称：
                      # `my_custom_script.py:my_func`] 使用数据加载器脚本时，
                      # 这指定加载数据的函数名称。默认为 `load_data`。
```

### TensorRT 构建器配置 (TensorRT Builder Configuration)
```bash
--trt-min-shapes TRT_MIN_SHAPES [TRT_MIN_SHAPES ...]
                      # 优化配置文件将支持的最小形状。为每个配置文件指定一次此选项。
                      # 如果未提供，将使用推理时输入形状。
                      # 格式：--trt-min-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]
--trt-opt-shapes TRT_OPT_SHAPES [TRT_OPT_SHAPES ...]
                      # 优化配置文件最高性能的形状。为每个配置文件指定一次此选项。
                      # 如果未提供，将使用推理时输入形状。
                      # 格式：--trt-opt-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]
--trt-max-shapes TRT_MAX_SHAPES [TRT_MAX_SHAPES ...]
                      # 优化配置文件将支持的最大形状。为每个配置文件指定一次此选项。
                      # 如果未提供，将使用推理时输入形状。
                      # 格式：--trt-max-shapes <input0>:[D0,D1,..,DN] .. <inputN>:[D0,D1,..,DN]
--tf32                # 在 TensorRT 中启用 tf32 精度
--fp16                # 在 TensorRT 中启用 fp16 精度
--bf16                # 在 TensorRT 中启用 bf16 精度
--fp8                 # 在 TensorRT 中启用 fp8 精度
--int8                # 在 TensorRT 中启用 int8 精度。如果需要校准但未提供校准缓存，
                      # 此选项将导致 TensorRT 使用 Polygraphy 数据加载器
                      # 提供校准数据来运行 int8 校准。如果运行校准且模型具有动态形状，
                      # 将使用最后一个优化配置文件作为校准配置文件。
--precision-constraints {prefer,obey,none}
                      # 如果设置为 `prefer`，TensorRT 将限制可用策略为网络中指定的层精度，
                      # 除非不存在首选层约束的实现，在这种情况下将发出警告并使用最快的可用实现。
                      # 如果设置为 `obey`，如果不存在首选层约束的实现，TensorRT 将构建网络失败。
                      # 默认为 `none`
--sparse-weights      # 在 TensorRT 中启用稀疏权重优化
--version-compatible  # 构建设计为前向 TensorRT 版本兼容的引擎。
--exclude-lean-runtime
                      # 启用版本兼容性时从计划中排除精简运行时。
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
                      # 加载策略计时缓存的路径。用于缓存策略计时信息以加速引擎构建过程。
                      # 如果 --load-timing-cache 指定的文件不存在，Polygraphy 将发出警告并
                      # 回退到使用空计时缓存。
--error-on-timing-cache-miss
                      # 当计时缓存中不存在正在计时的策略时发出错误。
--disable-compilation-cache
                      # 禁用缓存 JIT 编译代码
--save-tactics SAVE_TACTICS, --save-tactic-replay SAVE_TACTICS
                      # 保存 Polygraphy 策略重放文件的路径。
                      # TensorRT 选择的策略详细信息将被记录并作为 JSON 文件存储在此位置。
--load-tactics LOAD_TACTICS, --load-tactic-replay LOAD_TACTICS
                      # 加载 Polygraphy 策略重放文件的路径，例如由 --save-tactics 创建的文件。
                      # 文件中指定的策略将用于覆盖 TensorRT 的默认选择。
--tactic-sources [TACTIC_SOURCES ...]
                      # 要启用的策略源。这控制 TensorRT 允许从哪些库（例如 cudnn、cublas 等）
                      # 加载策略。值来自 trt.TacticSource 枚举中值的名称，不区分大小写。
                      # 如果未提供参数，例如 '--tactic-sources'，则禁用所有策略源。
                      # 默认为 TensorRT 的默认策略源。
--trt-config-script TRT_CONFIG_SCRIPT
                      # 定义创建 TensorRT IBuilderConfig 函数的 Python 脚本路径。
                      # 函数应接受构建器和网络作为参数并返回 TensorRT 构建器配置。
                      # 指定此选项时，将忽略所有其他配置参数。
                      # 默认情况下，Polygraphy 查找名为 `load_config` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。
                      # 例如：`my_custom_script.py:my_func`
--trt-config-func-name TRT_CONFIG_FUNC_NAME
                      # [已弃用 - 可以使用 --trt-config-script 指定函数名称：
                      # `my_custom_script.py:my_func`] 使用 trt-config-script 时，
                      # 这指定创建配置的函数名称。默认为 `load_config`。
--trt-config-postprocess-script TRT_CONFIG_POSTPROCESS_SCRIPT, --trt-cpps TRT_CONFIG_POSTPROCESS_SCRIPT
                      # [实验性] 定义修改 TensorRT IBuilderConfig 函数的 Python 脚本路径。
                      # 此函数将在 Polygraphy 完成创建构建器配置后调用，
                      # 应接受构建器、网络和配置作为参数并就地修改配置。
                      # 与 `--trt-config-script` 不同，所有其他配置参数将反映在传递给函数的配置中。
                      # 默认情况下，Polygraphy 查找名为 `postprocess_config` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。
                      # 例如：`my_custom_script.py:my_func`
--trt-safety-restricted
                      # 在 TensorRT 中启用安全范围检查
--refittable          # 使引擎能够在构建后用新权重重新拟合。
--strip-plan          # 构建时剥离可重新拟合权重的引擎。
--use-dla             # [实验性] 使用 DLA 作为默认设备类型
--allow-gpu-fallback  # [实验性] 允许 DLA 不支持的层回退到 GPU。
                      # 如果未设置 --use-dla 则无效果。
--pool-limit MEMORY_POOL_LIMIT [MEMORY_POOL_LIMIT ...], --memory-pool-limit MEMORY_POOL_LIMIT [MEMORY_POOL_LIMIT ...]
                      # 内存池限制。内存池名称来自 `trt.MemoryPoolType` 枚举中值的名称，
                      # 不区分大小写。格式：`--pool-limit <pool_name>:<pool_limit> ...`。
                      # 例如，`--pool-limit dla_local_dram:1e9 workspace:16777216`。
                      # 可选择使用 `K`、`M` 或 `G` 后缀来表示 KiB、MiB 或 GiB。
                      # 例如，`--pool-limit workspace:16M` 等于 `--pool-limit workspace:16777216`。
--preview-features [PREVIEW_FEATURES ...]
                      # 要启用的预览功能。值来自 trt.PreviewFeature 枚举中值的名称，
                      # 不区分大小写。如果未提供参数，例如 '--preview-features'，
                      # 则禁用所有预览功能。默认为 TensorRT 的默认预览功能。
--builder-optimization-level BUILDER_OPTIMIZATION_LEVEL
                      # 构建器优化级别。设置更高的优化级别允许优化器花费更多时间
                      # 寻找优化机会。与使用较低优化级别构建的引擎相比，
                      # 生成的引擎可能具有更好的性能。详见 TensorRT API 文档。
--hardware-compatibility-level HARDWARE_COMPATIBILITY_LEVEL
                      # 用于引擎的硬件兼容性级别。这允许在一种 GPU 架构上构建的引擎
                      # 在其他架构的 GPU 上工作。值来自 `trt.HardwareCompatibilityLevel` 
                      # 枚举中值的名称，不区分大小写。
                      # 例如，`--hardware-compatibility-level ampere_plus`
--max-aux-streams MAX_AUX_STREAMS
                      # TensorRT 允许使用的辅助流的最大数量。
                      # 如果网络包含可以并行运行的操作符，TRT 可以使用辅助流
                      # 以及提供给 IExecutionContext.execute_async_v3() 调用的流来执行它们。
                      # 辅助流的默认最大数量由 TensorRT 中的启发式确定，
                      # 基于启用多流是否会改善性能。详见 TensorRT API 文档。
--quantization-flags [QUANTIZATION_FLAGS ...]
                      # 要启用的 Int8 量化标志。值来自 trt.QuantizationFlag 枚举中值的名称，
                      # 不区分大小写。如果未提供参数，例如 '--quantization-flags'，
                      # 则禁用所有量化标志。默认为 TensorRT 的默认量化标志。
--profiling-verbosity PROFILING_VERBOSITY
                      # 生成引擎中 NVTX 注释的详细程度。
                      # 值来自 `trt.ProfilingVerbosity` 枚举中值的名称，不区分大小写。
                      # 例如，`--profiling-verbosity detailed`。默认为 'detailed'。
--weight-streaming    # 构建权重可流式传输的引擎。必须与 --strongly-typed 一起设置。
                      # 权重流式传输量可以通过 --weight-streaming-budget 设置。
--runtime-platform RUNTIME_PLATFORM
                      # TensorRT 引擎执行的目标运行时平台（操作系统和 CPU 架构）。
                      # 当目标运行时平台与构建平台不同时，TensorRT 提供跨平台引擎兼容性支持。
--engine-capability ENGINE_CAPABILITY
                      # 引擎功能。值来自 `trt.EngineCapability` 枚举中值的名称，不区分大小写。
--direct-io           # 如果设置，则引擎将直接从用户缓冲区读取和写入。
--tiling-optimization-level TILING_OPTIMIZATION_LEVEL
                      # 启用 TensorRT 中的瓦片层融合的优化级别。该功能是实验性的。
--plugins PLUGINS [PLUGINS ...]
                      # 要加载的插件库路径。每个路径应指向共享库（.so 文件）。
--trt-outputs TRT_OUTPUTS [TRT_OUTPUTS ...]
                      # 要标记为输出的 TensorRT 张量的名称。
                      # 使用特殊值 'mark all' 表示所有张量都应用作输出
--trt-exclude-outputs TRT_EXCLUDE_OUTPUTS [TRT_EXCLUDE_OUTPUTS ...]
                      # [实验性] 要取消标记为输出的 TensorRT 输出的名称。
--layer-precisions LAYER_PRECISIONS [LAYER_PRECISIONS ...]
                      # 为各层指定计算精度。格式：--layer-precisions <layer_name>:<precision> ...
                      # 例如：--layer-precisions example_layer:fp16 other_layer:int8
                      # 精度值来自 DataType 枚举中值的名称，不区分大小写。
                      # 例如：float, half, int8
--tensor-dtypes TENSOR_DTYPES [TENSOR_DTYPES ...]
                      # 为网络中的张量指定数据类型。格式：--tensor-dtypes <tensor_name>:<dtype> ...
                      # 例如：--tensor-dtypes example_tensor:fp16 other_tensor:int8
                      # 数据类型来自 DataType 枚举中值的名称，不区分大小写。
                      # 例如：float, half, int8
--tensor-formats TENSOR_FORMATS [TENSOR_FORMATS ...]
                      # 为网络中的张量指定格式。格式：--tensor-formats <tensor_name>:<format> ...
                      # 例如：--tensor-formats example_tensor:chw4 other_tensor:hwc8
                      # 张量格式来自 TensorFormat 枚举中值的名称，不区分大小写。
--trt-network-func-name TRT_NETWORK_FUNC_NAME
                      # [已弃用 - 函数名可以用 model_file 指定：`my_custom_script.py:my_func`]
                      # 使用 trt-network-script 时，这指定加载网络的函数名称。默认为 `load_network`。
--trt-network-postprocess-script TRT_NETWORK_POSTPROCESS_SCRIPT [TRT_NETWORK_POSTPROCESS_SCRIPT ...]
                      # [实验性] 定义修改 TensorRT 网络函数的 Python 脚本路径。
                      # 每个函数应接受构建器、网络和解析器（可能为 None）作为参数并就地修改网络。
                      # 默认情况下，Polygraphy 查找名为 `postprocess_network` 的函数。
                      # 您可以通过用冒号分隔来指定自定义函数名称。
                      # 例如：`my_custom_script.py:my_func`
--strongly-typed      # 创建强类型网络。在强类型网络中，类型信息和张量格式是必需的，
                      # 将进行类型和格式传播，并执行类型检查规则。
--mark-debug MARK_DEBUG [MARK_DEBUG ...]
                      # 要标记为调试张量的张量名称，以便它们可以作为引擎的输出。
--mark-unfused-tensors-as-debug-tensors
                      # 将所有非融合张量标记为调试张量，以便它们可以作为引擎的输出。
--save-timing-cache SAVE_TIMING_CACHE
                      # 保存策略计时缓存的路径。
--onnx-flags ONNX_FLAGS [ONNX_FLAGS ...]
                      # 要在 ONNX-Runtime 中设置的标志。格式：--onnx-flags <flag_name>:<flag_value> ...
                      # 例如：--onnx-flags enable_cpu_mem_arena:0
--plugin-instancenorm # [已弃用] 
```

### ONNX-Runtime 会话创建 (ONNX-Runtime Session Creation)
```bash
--providers PROVIDERS [PROVIDERS ...], --execution-providers PROVIDERS [PROVIDERS ...]
                      # 按优先级顺序使用的执行提供程序列表。
                      # 每个提供程序可以是完全匹配或不区分大小写的部分匹配，
                      # 用于 ONNX-Runtime 中可用的执行提供程序。
                      # 例如，'cpu' 值将匹配 'CPUExecutionProvider'
```

## 💡 使用示例

### 1. 基础转换
```bash
# ONNX 转 TensorRT
polygraphy convert model.onnx --convert-to trt -o model.engine

# 指定模型类型
polygraphy convert model.onnx --model-type onnx --convert-to trt -o model.engine
```

### 2. 精度优化转换
```bash
# FP16 转换
polygraphy convert model.onnx --convert-to trt --fp16 -o model_fp16.engine

# INT8 转换（需要校准缓存）
polygraphy convert model.onnx --convert-to trt --int8 --calibration-cache calib.cache -o model_int8.engine

# 混合精度
polygraphy convert model.onnx --convert-to trt --fp16 --tf32 -o model_mixed.engine
```

### 3. 动态形状转换
```bash
# 单个优化配置文件
polygraphy convert model.onnx --convert-to trt \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[4,3,224,224] \
  --trt-max-shapes input:[8,3,224,224] \
  -o dynamic_model.engine

# 多个输入
polygraphy convert model.onnx --convert-to trt \
  --trt-min-shapes image:[1,3,224,224] labels:[1,1000] \
  --trt-opt-shapes image:[4,3,224,224] labels:[4,1000] \
  --trt-max-shapes image:[8,3,224,224] labels:[8,1000] \
  -o multi_input.engine
```

### 4. 外部数据处理
```bash
# 指定外部数据目录
polygraphy convert large_model.onnx --convert-to trt \
  --external-data-dir ./weights/ -o large_model.engine

# 保存外部数据
polygraphy convert model.onnx --convert-to onnx \
  --save-external-data external_data.bin \
  --external-data-size-threshold 1M \
  -o model_with_external.onnx
```

### 5. 高级TensorRT配置
```bash
# 内存池限制
polygraphy convert model.onnx --convert-to trt \
  --pool-limit workspace:2G dla_local_dram:1G \
  -o memory_optimized.engine

# 策略控制
polygraphy convert model.onnx --convert-to trt \
  --save-tactics tactics.json \
  --tactic-sources cudnn cublas \
  -o model_with_tactics.engine

# 硬件兼容性
polygraphy convert model.onnx --convert-to trt \
  --hardware-compatibility-level ampere_plus \
  --version-compatible \
  -o compatible_model.engine
```

### 6. 调试和分析
```bash
# 标记调试张量
polygraphy convert model.onnx --convert-to trt \
  --mark-debug conv1_output relu2_output \
  -o debug_model.engine

# 详细日志
polygraphy convert model.onnx --convert-to trt \
  --verbose --log-format timestamp line-info \
  --log-file conversion.log \
  -o model.engine
```

## 📚 相关文档

- [run - 跨框架比较](./run.md) - 验证转换后模型的精度
- [inspect - 模型分析](./inspect.md) - 分析转换前后的模型差异
- [check - 模型验证](./check.md) - 转换前验证模型完整性

---

*`polygraphy convert` 支持多种模型格式转换，特别是针对 TensorRT 的优化转换提供了丰富的配置选项。*