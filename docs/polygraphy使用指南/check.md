# polygraphy check - 检查和验证模型

`polygraphy check` 提供模型检查和验证功能，用于检查和验证模型的各个方面。

## 📋 基本语法

```bash
polygraphy check [-h] [-v] [-q] [--verbosity VERBOSITY [VERBOSITY ...]]
                 [--silent]
                 [--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]]
                 [--log-file LOG_FILE]
                 {lint} ...
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
                      # 详细程度值应来自 Logger 类中定义的 Polygraphy 
                      # 日志详细程度，不区分大小写。
                      # 例如：`--verbosity INFO` 或 `--verbosity verbose`
                      # 要指定每个路径的详细程度，使用格式：
                      # `<path>:<verbosity>`。
                      # 例如：`--verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE`
                      # 路径应相对于 `polygraphy/` 目录。
                      # 使用最接近匹配的路径来确定详细程度。
--silent              # 禁用所有输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
                      # 日志消息格式：
                      # {'timestamp': 包含时间戳, 'line-info': 包含文件和行号, 'no-colors': 禁用颜色}
--log-file LOG_FILE   # Polygraphy 日志输出应写入的文件路径。
                      # 这可能不包括来自依赖项（如 TensorRT 或 ONNX-Runtime）的日志输出。
```

## 🔍 子命令

### lint - ONNX 模型拓扑检查

`[EXPERIMENTAL] 拓扑"检查"ONNX模型以查找图中的错误节点。所有依赖于错误节点的节点都将被标记为错误并被忽略。`

#### 基本用法
```bash
# 基础模型检查
polygraphy check lint model.onnx

# 保存检查报告
polygraphy check lint model.onnx -o report.json

# 详细检查日志
polygraphy check lint model.onnx --verbose
```

#### 完整参数列表

##### 位置参数
```bash
model_file            # 模型路径
```

##### 可选参数
```bash
-h, --help            # 显示此帮助信息并退出
-o OUTPUT, --output OUTPUT
                      # 保存 json 报告的路径
```

##### 日志选项 (Logging)
```bash
-v, --verbose         # 增加日志详细程度。可多次指定以获得更高详细程度
-q, --quiet           # 降低日志详细程度。可多次指定以获得更低详细程度
--verbosity VERBOSITY [VERBOSITY ...]
                      # 日志详细程度设置（同全局选项）
--silent              # 禁用所有输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
                      # 日志消息格式设置
--log-file LOG_FILE   # 日志文件输出路径
```

##### 模型选项 (Model)
```bash
--input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --inputs INPUT_SHAPES [INPUT_SHAPES ...]
                      # 模型输入及其形状。用于确定在为推理生成输入数据时使用的形状。
                      # 格式：--input-shapes <name>:<shape>
                      # 例如：--input-shapes image:[1,3,224,224] other_input:[10]
```

##### ONNX 模型加载选项 (ONNX Model Loading)
```bash
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
                      # 包含模型外部数据的目录路径。
                      # 通常，只有在外部数据未存储在模型目录中时才需要此选项。
--ignore-external-data
                      # 忽略外部数据，仅加载模型结构而不加载任何权重。
                      # 该模型仅可用于不需要权重的目的，例如提取子图或检查模型结构。
                      # 在外部数据不可用的情况下，这可能很有用。
--fp-to-fp16          # 将 ONNX 模型中的所有浮点张量转换为 16 位精度。
                      # 这不是使用 TensorRT fp16 精度所必需的，但对于其他后端可能有用。
                      # 需要 onnxmltools。
```

##### 数据加载器选项 (Data Loader)
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

##### ONNX-Runtime 会话创建选项 (ONNX-Runtime Session Creation)
```bash
--providers PROVIDERS [PROVIDERS ...], --execution-providers PROVIDERS [PROVIDERS ...]
                      # 按优先级顺序使用的执行提供程序列表。
                      # 每个提供程序可以是完全匹配或不区分大小写的部分匹配，
                      # 用于 ONNX-Runtime 中可用的执行提供程序。
                      # 例如，'cpu' 值将匹配 'CPUExecutionProvider'
```

## 💡 使用示例

### 1. 基础模型检查
```bash
# 简单检查
polygraphy check lint model.onnx

# 详细检查
polygraphy check lint model.onnx --verbose

# 保存检查报告
polygraphy check lint model.onnx -o lint_report.json
```

### 2. 指定输入形状检查
```bash
# 为动态形状模型指定具体形状
polygraphy check lint model.onnx --input-shapes input:[1,3,224,224]

# 多个输入
polygraphy check lint model.onnx --input-shapes \
  image:[1,3,224,224] \
  labels:[1,1000]
```

### 3. 外部数据模型检查
```bash
# 指定外部数据目录
polygraphy check lint model.onnx --external-data-dir ./external_weights/

# 忽略外部数据（仅检查结构）
polygraphy check lint model.onnx --ignore-external-data
```

### 4. 使用自定义数据检查
```bash
# 使用预定义输入数据
polygraphy check lint model.onnx --load-inputs test_data.json

# 使用数据加载脚本
polygraphy check lint model.onnx --data-loader-script data_loader.py

# 指定随机数种子
polygraphy check lint model.onnx --seed 42 --iterations 5
```

### 5. 精度转换检查
```bash
# 转换为FP16并检查
polygraphy check lint model.onnx --fp-to-fp16
```

## 📊 JSON 报告格式

lint 子命令生成的 JSON 报告包含以下字段：

```json
{
    "summary": {
        "passing": ["<通过 ORT 推理检查的节点列表>"],
        "failing": ["<未通过 ORT 推理检查的节点列表>"]
    },
    "lint_entries": [
        {
            "level": "<严重程度级别>",
            "source": "<错误来源>",
            "message": "<错误字符串>",
            "nodes": ["<失败节点名称>"]
        }
    ]
}
```

### 报告字段说明
- **summary**: 汇总通过和失败的节点
  - **passing**: 通过检查的节点列表
  - **failing**: 未通过检查的节点列表
- **lint_entries**: 检查条目列表
  - **level**: 严重程度（error 或 warning）
  - **source**: 生成错误消息的底层检查器（`onnx.checker` 或 ONNX Runtime）
  - **message**: 错误消息
  - **nodes**: 与错误消息相关的节点列表（可选）

## ⚠️ 已知限制

1. **数据类型支持**: 目前不支持 BFLOAT16 和 FLOAT8
2. **错误节点检测**: 只捕获相互独立的错误节点，不检查依赖于错误节点的下游节点
3. **子图限制**: 不递归检查节点内嵌套的子图
4. **自定义操作**: 自定义操作在 JSON 报告中记录为警告，但被内部推理检查视为异常
5. **数据依赖性**: 子工具基于用户输入数据或为输入张量生成随机数据来验证数据相关故障，因此覆盖范围完全取决于输入数据
6. **大模型限制**: 大模型（>2GB）要求外部数据与模型文件在同一目录中，不支持外部数据的自定义路径

## 💡 实用脚本

### 批量模型检查脚本
```bash
#!/bin/bash
# batch_lint.sh

models_dir="models"
report_dir="lint_reports_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$report_dir"

echo "开始批量检查模型..."

for model in "$models_dir"/*.onnx; do
    if [[ -f "$model" ]]; then
        model_name=$(basename "$model" .onnx)
        echo "检查模型: $model_name"
        
        polygraphy check lint "$model" \
          -o "$report_dir/${model_name}_lint_report.json" \
          --verbose \
          > "$report_dir/${model_name}_lint.log" 2>&1
        
        if [[ $? -eq 0 ]]; then
            echo "✅ $model_name 检查通过"
        else
            echo "❌ $model_name 检查失败"
        fi
    fi
done

echo "批量检查完成，报告保存在: $report_dir/"
```

### 检查报告分析脚本
```python
#!/usr/bin/env python3
# analyze_lint_reports.py

import json
import sys
from pathlib import Path

def analyze_lint_report(report_path):
    """分析单个lint报告"""
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"无法读取报告 {report_path}: {e}"
    
    summary = report.get('summary', {})
    lint_entries = report.get('lint_entries', [])
    
    passing_count = len(summary.get('passing', []))
    failing_count = len(summary.get('failing', []))
    
    error_count = len([entry for entry in lint_entries if entry.get('level') == 'error'])
    warning_count = len([entry for entry in lint_entries if entry.get('level') == 'warning'])
    
    result = {
        'report_path': report_path,
        'passing_nodes': passing_count,
        'failing_nodes': failing_count,
        'errors': error_count,
        'warnings': warning_count,
        'status': 'PASS' if error_count == 0 else 'FAIL'
    }
    
    return result

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_lint_reports.py <报告目录或文件>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # 分析单个报告文件
        result = analyze_lint_report(path)
        if isinstance(result, dict):
            print(f"报告: {result['report_path']}")
            print(f"状态: {result['status']}")
            print(f"通过节点: {result['passing_nodes']}")
            print(f"失败节点: {result['failing_nodes']}")
            print(f"错误: {result['errors']}")
            print(f"警告: {result['warnings']}")
        else:
            print(result)
    elif path.is_dir():
        # 批量分析报告目录
        report_files = list(path.glob("*_lint_report.json"))
        
        if not report_files:
            print(f"在目录 {path} 中未找到lint报告文件")
            sys.exit(1)
        
        print(f"发现 {len(report_files)} 个报告文件\n")
        
        all_results = []
        for report_file in sorted(report_files):
            result = analyze_lint_report(report_file)
            if isinstance(result, dict):
                all_results.append(result)
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                print(f"{status_icon} {report_file.name}: {result['status']} "
                      f"(错误: {result['errors']}, 警告: {result['warnings']})")
            else:
                print(f"❌ {report_file.name}: {result}")
        
        # 汇总统计
        if all_results:
            total_reports = len(all_results)
            passed_reports = len([r for r in all_results if r['status'] == 'PASS'])
            total_errors = sum(r['errors'] for r in all_results)
            total_warnings = sum(r['warnings'] for r in all_results)
            
            print(f"\n📊 汇总统计:")
            print(f"总报告数: {total_reports}")
            print(f"通过数: {passed_reports}")
            print(f"失败数: {total_reports - passed_reports}")
            print(f"总错误数: {total_errors}")
            print(f"总警告数: {total_warnings}")
    else:
        print(f"路径 {path} 不存在")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📚 相关文档

- [convert - 模型格式转换](./convert.md) - 转换前检查模型完整性
- [run - 跨框架比较](./run.md) - 验证检查后的模型推理结果
- [inspect - 模型分析](./inspect.md) - 深入分析检查发现的问题

---

*`polygraphy check lint` 是模型验证的重要工具，建议在模型转换和部署前进行检查以发现潜在问题。*