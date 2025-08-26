# 手动处理运行结果和保存的输入

## 简介

来自 `Comparator.run` 的推理输入和输出可以被序列化并保存到 JSON 文件中，以便可以重复使用。输入存储为 `List[Dict[str, np.ndarray]]`，而输出存储在 `RunResults` 对象中，该对象可以跟踪来自多个推理迭代的多个运行器的输出。

提供 `--save-inputs` 和 `--save-outputs` 选项的命令行工具通常使用这些格式。

通常，您只会将保存的输入或 `RunResults` 与其他 Polygraphy API 或工具一起使用(如[此示例](../../cli//run/06_comparing_with_custom_output_data/)或[此示例](../../cli/inspect/05_inspecting_inference_outputs/))，但有时，您可能希望手动处理底层的 NumPy 数组。

Polygraphy 包含一些方便的 API，可以轻松加载和操作这些对象。

此示例说明了如何使用 Python API 从文件中加载保存的输入和/或 `RunResults`，然后访问其中存储的 NumPy 数组。

## 运行示例

1.  生成一些推理输入和输出：

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --save-inputs inputs.json --save-outputs outputs.json
    ```

2.  **[可选]** 使用 `inspect data` 在命令行上查看输入：

    ```bash
    polygraphy inspect data inputs.json --show-values
    ```

3.  **[可选]** 使用 `inspect data` 在命令行上查看输出：

    ```bash
    polygraphy inspect data outputs.json --show-values
    ```

4.  运行示例：

    ```bash
    python3 example.py
    ```
