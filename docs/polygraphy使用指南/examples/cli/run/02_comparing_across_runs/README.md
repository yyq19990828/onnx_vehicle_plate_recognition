# 跨运行比较

## 先决条件
有关如何使用 `polygraphy run` 比较不同框架输出的一般概述，请参阅[比较框架](../../../../examples/cli/run/01_comparing_frameworks)的示例。

## 简介

在某些情况下，您可能需要比较 `polygraphy run` 命令的不同调用之间的结果。其中一些示例包括：

*   比较不同平台之间的结果
*   比较不同版本的 TensorRT 之间的结果
*   比较具有兼容输入/输出的不同模型类型

在本示例中，我们将演示如何使用 Polygraphy 完成此操作。

## 运行示例

### 跨运行比较

1.  保存第一次运行的输入和输出值：

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-inputs inputs.json --save-outputs run_0_outputs.json
    ```

2.  再次运行模型，这次加载第一次运行中保存的输入和输出。保存的输入将用作当前运行的输入，保存的输出将用于与第一次运行进行比较。

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```

    `--atol`/`--rtol`/`--check-error-stat` 选项的工作方式与[比较框架](../../../../examples/cli/run/01_comparing_frameworks)示例中的相同：

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json \
        --atol 0.001 --rtol 0.001 --check-error-stat median
    ```

### 比较不同的模型

我们还可以使用此技术来比较不同的模型，例如 TensorRT 引擎和 ONNX 模型（如果它们具有匹配的输出）。

1.  将 ONNX 模型转换为 TensorRT 引擎并将其保存到磁盘：

    ```bash
    polygraphy convert identity.onnx -o identity.engine
    ```

2.  在 Polygraphy 中运行保存的引擎，使用 ONNX-Runtime 运行中保存的输入作为引擎的输入，并将引擎的输出与保存的 ONNX-Runtime 输出进行比较：

    ```bash
    polygraphy run --trt identity.engine --model-type=engine \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```


## 进一步阅读

有关如何使用 Python API 访问和使用已保存输出的详细信息，请参阅 [API 示例 08](../../../api/08_working_with_run_results_and_saved_inputs_manually/)。

有关与自定义输出进行比较的信息，请参阅 [`run` 示例 06](../06_comparing_with_custom_output_data/)。
