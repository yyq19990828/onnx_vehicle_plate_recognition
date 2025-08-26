# polygraphy debug - 调试工具集

[EXPERIMENTAL] 调试各种模型问题。

## 📋 基本工作原理

`debug` 子工具基于以下通用原则工作：

1. **迭代执行任务** - 生成某些输出
2. **评估输出** - 确定输出应被视为 `good` 或 `bad`
3. **工件分类** - 根据(2)将任何跟踪的工件分类到 `good` 和 `bad` 目录
4. **迭代调整** - 根据需要进行更改，然后重复该过程

步骤(1)中提到的"某些输出"通常是模型文件，默认在每次迭代期间写入当前目录。

为了区分 `good` 和 `bad`，子工具使用以下两种方法之一：
- **`--check` 命令**：如果提供，可以是几乎任何命令，这使得 `debug` 极其灵活
- **交互式提示**：如果未提供 `--check` 命令，子工具将以交互方式提示您报告迭代是通过还是失败

可以使用 `--artifacts` 指定要跟踪的每次迭代工件。当迭代失败时，它们被移动到 `bad` 目录，否则移动到 `good` 目录。工件可以是任何文件或目录。例如，这可用于排序日志或 TensorRT 策略重放文件，甚至是每次迭代的输出（通常是 TensorRT 引擎或 ONNX 模型）。

默认情况下，如果 `--check` 命令的状态码非零，则该迭代被视为失败。您可以选择使用其他命令行选项以更精细的方式控制失败的定义：
- `--fail-regex` 允许您仅在 `--check` 的输出（在 `stdout` 或 `stderr` 上）匹配一个或多个正则表达式时计算失败，并忽略任何其他错误
- `--fail-returncode` 让您指定要计算为失败的状态码，排除所有其他非零状态码

## 📋 基本语法

```bash
polygraphy debug [-h] [-v] [-q] [--verbosity VERBOSITY [VERBOSITY ...]]
                 [--silent] [--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]]
                 [--log-file LOG_FILE]
                 {build,precision,reduce,repeat} ...
```

## ⚙️ 全局选项

### 帮助选项
```bash
-h, --help            # 显示帮助信息并退出
```

### 日志选项 (Logging)
```bash
-v, --verbose         # 增加日志详细程度。可多次指定以获得更高详细程度
-q, --quiet           # 降低日志详细程度。可多次指定以获得更低详细程度
--verbosity VERBOSITY [VERBOSITY ...]
                      # 要使用的日志详细程度。优先于 `-v` 和 `-q` 选项，
                      # 与它们不同，允许您控制每个路径的详细程度。
                      # 详细程度值应来自 Logger 类中定义的 Polygraphy 日志详细程度，不区分大小写。
                      # 例如：`--verbosity INFO` 或 `--verbosity verbose`。
                      # 要指定每个路径的详细程度，使用格式：`<path>:<verbosity>`。
                      # 例如：`--verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE`
                      # 路径应相对于 `polygraphy/` 目录。使用最接近匹配的路径来确定详细程度。
--silent              # 禁用所有输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
                      # 日志消息格式：
                      # {'timestamp': 包含时间戳, 'line-info': 包含文件和行号, 'no-colors': 禁用颜色}
--log-file LOG_FILE   # Polygraphy 日志输出应写入的文件路径。
                      # 这可能不包括来自依赖项（如 TensorRT 或 ONNX-Runtime）的日志输出。
```

## 🔧 子命令

### build - 重复构建引擎隔离错误策略

重复构建引擎以隔离有错误的策略。

具体执行以下操作：
1. 构建 TensorRT 引擎并默认保存在当前目录为 `polygraphy_debug.engine`
2. 使用 `--check` 命令（如果提供）或交互模式评估它
3. 根据(2)将 `--artifacts` 指定的文件分类到 `good` 和 `bad` 目录

#### 基本语法
```bash
polygraphy debug build model_file --until UNTIL [options...]
```

#### 关键参数

##### Pass/Fail 报告 (Pass/Fail Reporting)
```bash
--check ..., --check-inference ...
                      # 检查模型的命令。省略时启动交互调试会话。
                      # 默认退出状态 0 被视为'通过'，其他退出状态被视为'失败'。
--fail-code FAIL_CODES [FAIL_CODES ...], --fail-returncode FAIL_CODES [FAIL_CODES ...]
                      # 从 --check 命令计数为失败的返回码。
                      # 如果提供此选项，任何其他返回码将被计为成功。
--ignore-fail-code IGNORE_FAIL_CODES [IGNORE_FAIL_CODES ...]
                      # 从 --check 命令忽略为失败的返回码。
--fail-regex FAIL_REGEX [FAIL_REGEX ...]
                      # 表示检查命令输出中错误的正则表达式。
                      # 仅当在命令输出中找到匹配字符串时，命令才被视为失败。
--show-output         # 即使对于通过的迭代也显示 --check 命令的输出。
                      # 默认情况下，通过迭代的输出被捕获。
--hide-fail-output    # 抑制失败迭代的 --check 命令输出。
                      # 默认情况下，失败迭代的输出被显示。
```

##### 工件排序 (Artifact Sorting)
```bash
--artifacts ARTIFACTS [ARTIFACTS ...]
                      # 要排序的工件路径。这些将根据 `--check` 命令的退出状态
                      # 移动到 'good' 和 'bad' 目录，并添加迭代号、时间戳和返回码后缀。
--art-dir DIR, --artifacts-dir DIR
                      # 移动工件并将其分类到 'good' 和 'bad' 的目录。
                      # 默认为当前目录中名为 `polygraphy_artifacts` 的目录。
```

##### 迭代调试 (Iterative Debugging)
```bash
--iter-artifact ITER_ARTIFACT_PATH, --intermediate-artifact ITER_ARTIFACT_PATH
                      # 存储每次迭代中间工件的路径。默认为：polygraphy_debug.engine
--no-remove-intermediate
                      # 不要在迭代之间删除中间工件。
--iter-info ITERATION_INFO_PATH, --iteration-info ITERATION_INFO_PATH
                      # 写入包含当前迭代信息的 JSON 文件的路径。
--until UNTIL         # 控制何时停止运行。选择：['good', 'bad', int]。
                      # 'good' 将持续运行直到第一次'好'运行。
                      # 'bad' 将运行直到第一次'坏'运行。
                      # 可以指定整数来运行设定的迭代次数。
```

build 子命令还包括完整的 TensorRT 构建器配置选项（与 convert 命令相同）。

### precision - 精度调试

[EXPERIMENTAL] 迭代地标记层以在更高精度下运行，以找到性能和质量之间的折衷。

每次迭代将在当前目录中生成一个名为 'polygraphy_debug.engine' 的引擎。

#### 基本语法
```bash
polygraphy debug precision model_file [options...]
```

#### 精度特定选项
```bash
--mode {bisect,linear}
                      # 如何选择层以在更高精度下运行。
                      # 'bisect' 将使用二进制搜索，'linear' 将一次迭代标记一个额外的层
--dir {forward,reverse}, --direction {forward,reverse}
                      # 标记层以在更高精度下运行的顺序。
                      # 'forward' 将从网络输入开始标记层，'reverse' 将从网络输出开始
-p {float32,float16}, --precision {float32,float16}
                      # 标记层以在更高精度下运行时使用的精度
```

#### 调试重放 (Debug Replay)
```bash
--load-debug-replay LOAD_DEBUG_REPLAY
                      # 从中加载调试重放的路径。重放文件包含某些或所有迭代结果的信息，
                      # 允许您跳过这些迭代。
--save-debug-replay SAVE_DEBUG_REPLAY
                      # 保存调试重放的路径，其中包含调试迭代结果的信息。
                      # 重放可与 `--load-debug-replay` 一起使用以在后续调试会话期间跳过迭代。
                      # 默认为当前目录中的 `polygraphy_debug_replay.json`。
```

precision 子命令也包括模型加载、数据加载器和 TensorRT 配置选项。

### reduce - 模型缩减

[EXPERIMENTAL] 将失败的 ONNX 模型缩减为导致失败的最小节点集。

执行以下操作：
1. 生成给定 ONNX 模型的逐步较小子图，默认保存为 `polygraphy_debug.onnx`
2. 使用自动化方式（如果提供了 `--check` 命令）或交互方式评估它
3. 如果迭代失败，在后续迭代中进一步缩减模型；否则，扩展模型以包含原始模型中的更多节点
4. 当模型无法进一步缩减时，将其保存到 `--output` 指定的路径

#### 基本语法
```bash
polygraphy debug reduce model_file [options...]
```

#### 缩减特定选项
```bash
--min-good MIN_GOOD, --minimal-good MIN_GOOD
                      # 保存与缩减模型大小相近但没有失败的 ONNX 模型的路径。
                      # 不保证生成。
--no-reduce-inputs    # 不尝试更改图输入以进一步缩减模型。
                      # 'reduce' 将仅尝试找到最早的失败输出。
--no-reduce-outputs   # 不尝试更改图输出以进一步缩减模型。
                      # 'reduce' 将仅尝试找到最新的失败输入。
--mode {bisect,linear}
                      # 从模型中迭代删除节点的策略。
                      # 'bisect' 将使用二进制搜索，'linear' 将一次删除一个节点。
                      # 'linear' 模式可能明显较慢，但在有分支的模型中可以提供更好的结果。
```

#### 模型输入形状 (Model Input Shapes)
```bash
--model-input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --model-inputs INPUT_SHAPES [INPUT_SHAPES ...]
                      # 模型输入及其形状。用于确定在为推理生成输入数据时使用的形状。
                      # 格式：--model-input-shapes <name>:<shape>
                      # 例如：--model-input-shapes image:[1,3,224,224] other_input:[10]
```

#### ONNX 模型保存选项
```bash
-o SAVE_ONNX, --output SAVE_ONNX
                      # 保存 ONNX 模型的路径
--save-external-data [EXTERNAL_DATA_PATH], --external-data-path [EXTERNAL_DATA_PATH]
                      # 是否将权重数据保存在外部文件中。
--external-data-size-threshold EXTERNAL_DATA_SIZE_THRESHOLD
                      # 大小阈值（字节），超过此阈值的张量数据将存储在外部文件中。
--no-save-all-tensors-to-one-file
                      # 保存外部数据时不要将所有张量保存到单个文件中。
```

#### ONNX 形状推理选项
```bash
--no-shape-inference  # 加载模型时禁用 ONNX 形状推理
--force-fallback-shape-inference
                      # 强制 Polygraphy 使用 ONNX-Runtime 确定图中张量的元数据。
--no-onnxruntime-shape-inference
                      # 禁用使用 ONNX-Runtime 的形状推理工具。
```

### repeat - 重复执行命令

[EXPERIMENTAL] 重复运行任意命令，将生成的工件分类到 `good` 和 `bad` 目录。

这是最简单的调试子命令，主要用于重复执行和工件分类。

#### 基本语法
```bash
polygraphy debug repeat --until UNTIL [options...]
```

#### 必需参数
```bash
--until UNTIL         # 控制何时停止运行。选择：['good', 'bad', int]。
                      # 'good' 将持续运行直到第一次'好'运行。
                      # 'bad' 将运行直到第一次'坏'运行。
                      # 可以指定整数来运行设定的迭代次数。
```

repeat 子命令包括与其他调试子命令相同的日志、Pass/Fail 报告、工件排序和迭代调试选项。

## 💡 使用示例

### 1. 基础引擎构建调试
```bash
# 重复构建引擎直到找到第一个失败
polygraphy debug build model.onnx --until bad \
  --save-tactics tactics.json \
  --artifacts tactics.json

# 使用自定义检查命令
polygraphy debug build model.onnx --until 5 \
  --check "polygraphy run polygraphy_debug.engine --trt" \
  --artifacts polygraphy_debug.engine
```

### 2. 精度问题调试
```bash
# 二分法查找精度问题层
polygraphy debug precision model.onnx \
  --mode bisect --dir forward \
  --precision float32 \
  --check "python validate_precision.py"

# 线性方式逐层检查
polygraphy debug precision model.onnx \
  --mode linear --dir reverse \
  --precision float16
```

### 3. 模型缩减
```bash
# 基础模型缩减
polygraphy debug reduce model.onnx \
  -o reduced_model.onnx \
  --check "polygraphy run polygraphy_debug.onnx --onnxrt --trt"

# 指定输入形状的缩减
polygraphy debug reduce model.onnx \
  --model-input-shapes input:[1,3,224,224] \
  --mode bisect \
  -o minimal_failure.onnx

# 只缩减输出
polygraphy debug reduce model.onnx \
  --no-reduce-inputs \
  -o output_reduced.onnx
```

### 4. 重复测试
```bash
# 重复运行直到找到失败
polygraphy debug repeat --until bad \
  --check "python test_model.py" \
  --artifacts test_logs.txt

# 运行固定次数
polygraphy debug repeat --until 10 \
  --check "polygraphy run model.onnx --trt" \
  --artifacts engine_outputs.json
```

## 🔧 高级用法

### 自定义检查脚本示例
```python
#!/usr/bin/env python3
# check_model.py - 自定义模型检查脚本

import sys
import numpy as np
from polygraphy.backend.trt import TrtRunner

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "polygraphy_debug.engine"
    
    try:
        with TrtRunner(model_path) as runner:
            # 生成测试输入
            input_metadata = runner.get_input_metadata()
            inputs = {}
            for name, (dtype, shape) in input_metadata.items():
                inputs[name] = np.random.randn(*shape).astype(dtype)
            
            # 运行推理
            outputs = runner.infer(inputs)
            
            # 检查结果
            for name, output in outputs.items():
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    print(f"发现 NaN/Inf 在输出 {name}")
                    return 1  # 失败
            
            print("模型检查通过")
            return 0  # 成功
            
    except Exception as e:
        print(f"检查失败: {e}")
        return 1  # 失败

if __name__ == "__main__":
    sys.exit(main())
```

使用方式：
```bash
polygraphy debug build model.onnx --until bad \
  --check "python check_model.py"
```

### 工件排序工作流
```bash
# 1. 设置工件目录
mkdir debug_session
cd debug_session

# 2. 运行调试，收集多种工件
polygraphy debug build ../model.onnx --until 10 \
  --save-tactics tactics.json \
  --artifacts tactics.json polygraphy_debug.engine \
  --art-dir ./sorted_artifacts

# 3. 检查结果
ls sorted_artifacts/good/    # 成功的工件
ls sorted_artifacts/bad/     # 失败的工件
```

## ⚠️ 注意事项

1. **实验性功能**: 所有 debug 子工具都是实验性的，API 可能会改变
2. **资源消耗**: 调试过程可能很耗时，特别是对大模型使用 linear 模式
3. **存储空间**: 工件收集可能产生大量文件，注意磁盘空间
4. **检查命令**: 确保 `--check` 命令能正确区分成功和失败情况

## 📚 相关文档

- [run - 跨框架比较](./run.md) - 用于 --check 命令的好选择
- [convert - 模型转换](./convert.md) - 应用调试发现的修复方案
- [inspect - 模型分析](./inspect.md) - 分析调试过程中的工件

---

*`polygraphy debug` 工具集提供系统化的方法来定位和解决复杂的模型问题。*