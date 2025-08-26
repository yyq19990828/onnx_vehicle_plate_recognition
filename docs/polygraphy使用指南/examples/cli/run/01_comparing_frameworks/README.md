# 比较框架

## 简介

您可以使用 `run` 子工具来比较不同框架之间的模型。
在最简单的情况下，您可以提供一个模型和一个或多个框架标志。
默认情况下，它将生成综合输入数据，使用
指定的框架运行推理，然后比较指定框架的输出。

## 运行示例

在本示例中，我们将概述 `run` 子工具的各种常见用例：

- [比较 TensorRT 和 ONNX-Runtime 输出](#比较-tensorrt-和-onnx-runtime-输出)
- [比较 TensorRT 精度](#比较-tensorrt-精度)
- [更改容差](#更改容差)
- [更改比较指标](#更改比较指标)
- [比较 ONNX-Runtime 和 TensorRT 之间的逐层输出](#比较-onnx-runtime-和-tensorrt-之间的逐层输出)

### 比较 TensorRT 和 ONNX-Runtime 输出

要在 Polygraphy 中使用这两个框架运行模型并执行输出
比较：

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt
```

`dynamic_identity.onnx` 模型具有动态输入形状。默认情况下，
Polygraphy 会将模型中的任何动态输入维度覆盖为
`constants.DEFAULT_SHAPE_VALUE`（定义为 `1`）并向您发出警告：

<!-- Polygraphy Test: Ignore Start -->
```
[W]     输入张量：X (dtype=DataType.FLOAT, shape=(1, 2, -1, -1)) | 未提供形状；将对配置文件中的 min/opt/max 使用形状：[1, 2, 1, 1]。
[W]     这将导致张量具有静态形状。如果这是不正确的，请设置此输入张量的形状范围。
```
<!-- Polygraphy Test: Ignore End -->

为了抑制此消息并明确向
Polygraphy 提供输入形状，请使用 `--input-shapes` 选项：

```
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]
```

### 比较 TensorRT 精度

要构建具有降低精度层的 TensorRT 引擎以与
ONNXRT 进行比较，请使用支持的精度标志之一（例如 `--tf32`、`--fp16`、`--int8` 等）。
例如：

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4]
```

> :warning: 使用 INT8 精度获得可接受的精度通常需要一个额外的校准步骤：
  请参阅[开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
  和有关[如何在命令行上使用 Polygraphy 进行校准](../../../../examples/cli/convert/01_int8_calibration_in_tensorrt)
  的说明。

### 更改容差

`run` 使用的默认容差通常适用于 FP32 精度
但可能不适用于降低的精度。为了放宽容差，
您可以使用 `--atol` 和 `--rtol` 选项来分别设置绝对和相对
容差。

### 更改比较指标

您可以使用 `--check-error-stat` 选项来更改用于
比较的指标。默认情况下，Polygraphy 使用“逐元素”指标
(`--check-error-stat elemwise`)。

`--check-error-stat` 的其他可能指标是 `mean`、`median` 和 `max`，它们
分别比较张量上的平均、中位数和最大绝对/相对误差。

为了更好地理解这一点，假设我们正在
比较两个输出 `out0` 和 `out1`。Polygraphy 取
这些张量的逐元素绝对和相对差异：

<!-- Polygraphy Test: Ignore Start -->
```
absdiff = out0 - out1
reldiff = absdiff / abs(out1)
```
<!-- Polygraphy Test: Ignore End -->

然后，对于输出中的每个索引 `i`，Polygraphy 检查
`absdiff[i] > atol and reldiff[i] > rtol` 是否成立。如果任何索引满足此条件，
则比较将失败。这不如比较
整个张量上的最大绝对和相对误差 (`--check-error-stat max`) 严格，因为如果
*不同的*索引 `i` 和 `j` 满足 `absdiff[i] > atol` 和 `reldiff[j] > rtol`，
则 `max` 比较将失败，但 `elemwise` 比较可能
会通过。

综上所述，以下示例在使用 FP16 的 TensorRT 和 ONNX-Runtime 之间运行 `median` 比较，
使用 `0.001` 的绝对和相对容差：

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4] \
    --atol 0.001 --rtol 0.001 --check-error-stat median
```

> 您还可以为 `--atol`/`--rtol`/`--check-error-stat` 指定每个输出的值。
  有关更多信息，请参阅 `run` 子工具的帮助输出。

### 比较 ONNX-Runtime 和 TensorRT 之间的逐层输出

当网络输出不匹配时，比较逐层输出
以查看引入错误的位置可能很有用。为此，您可以分别使用 `--trt-outputs`
和 `--onnx-outputs` 选项。这些选项接受一个或多个
输出名称作为其参数。特殊值 `mark all` 表示应比较模型中的所有
张量：

```bash
 polygraphy run dynamic_identity.onnx --trt --onnxrt \
     --trt-outputs mark all \
     --onnx-outputs mark all
```

为了更容易地找到第一个不匹配的输出，您可以使用 `--fail-fast`
选项，这将导致工具在第一次输出不匹配后退出。

请注意，使用 `--trt-outputs mark all` 有时会由于计时、层融合选择和格式
约束的差异而扰乱生成的
引擎，这可能会隐藏失败。在这种情况下，您可能需要使用
更复杂的方法来二分失败的模型并生成一个重现错误的简化
测试用例。有关
如何使用 Polygraphy 执行此操作的教程，请参阅[减少失败的 ONNX 模型](../../../../examples/cli/debug/02_reducing_failing_onnx_models)。

## 进一步阅读

* 在某些情况下，您可能需要在多个 Polygraphy 运行中进行比较
  （例如，在比较预构建的 TensorRT 引擎或
  [Polygraphy 网络脚本](../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)
  的输出与 ONNX-Runtime 时）。有关如何
  完成此操作的教程，请参阅[跨运行比较](../../../../examples/cli/run/02_comparing_across_runs)。

* 有关在 TensorRT 中使用动态形状的更多详细信息：
  * 有关如何使用 Polygraphy CLI 为引擎指定
    优化配置文件的信息，请参阅[TensorRT 中的动态形状](../../../../examples/cli/convert/03_dynamic_shapes_in_tensorrt/)
  * 有关
    如何使用 Polygraphy API 执行此操作的详细信息，请参阅[TensorRT 和动态形状](../../../../examples/api/07_tensorrt_and_dynamic_shapes/)

* 有关如何提供真实输入数据的详细信息，请参阅[使用自定义输入数据进行比较](../05_comparing_with_custom_input_data/)。

* 有关如何使用 Polygraphy 调试精度失败的更广泛教程，请参阅[调试 TensorRT 精度问题](../../../../how-to/debug_accuracy.md)。
