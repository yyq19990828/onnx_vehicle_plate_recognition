# 添加精度约束

## 简介

当使用在 FP32 中训练的模型来构建利用
[降低精度优化](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision)的 TensorRT 引擎时，
模型中的某些层可能需要被约束为在 FP32 中运行以
保持可接受的精度。

以下示例演示了如何选择性地约束
网络中指定层的精度。提供的 ONNX 模型执行以下操作：

1.  通过右乘一个旋转 90 度的单位矩阵来水平翻转其输入，
2.  将 `FP16_MAX` 添加到翻转后的输入，然后从结果中减去 `FP16_MAX`，
3.  通过右乘旋转后的单位矩阵来水平翻转减法的结果。

如果 `x` 是正数，则此过程中的步骤 (2) 需要在 FP32 中完成才能
获得可接受的精度，因为值将超过 FP16 可表示的
范围（设计如此）。然而，当启用 FP16 优化而没有
约束时，TensorRT 由于不知道将为 `x` 使用什么范围的值，
通常会选择在此过程中的所有步骤中都以 FP16 运行：

*   步骤 (1) 和 (3) 中的 GEMM 操作在 FP16 中的运行速度将比在 FP32 中快
    （对于足够大的问题规模）

*   步骤 (2) 中的逐点操作在 FP16 中运行得更快，并且将
    数据保留在 FP16 中消除了对 FP32 之间额外重构的需要。

因此，您需要在 TensorRT 网络中约束允许的精度，
以便 TensorRT 在分配引擎中的层精度时做出适当的选择。

Polygraphy 命令行工具提供了多种约束层精度的方法：

1.  `--layer-precisions` 选项允许您为单个层设置精度。

2.  网络后处理脚本允许您以编程方式修改由 Polygraphy
    解析或以其他方式生成的 TensorRT 网络。

3.  网络加载器脚本允许您使用
    TensorRT Python API 手动构建整个 TensorRT 网络。在网络构建期间，您可以根据需要设置层精度。


## 运行示例

**警告：** _此示例需要 TensorRT 8.4 或更高版本。_

### 使用 `--layer-precisions` 选项

运行以下命令以比较使用 FP16
优化的 TensorRT 运行模型与在 FP32 中运行的 ONNX-Runtime：

<!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy run needs_constraints.onnx \
    --trt --fp16 --onnxrt --val-range x:[1,2] \
    --layer-precisions Add:float16 Sub:float32 --precision-constraints prefer \
    --check-error-stat median
```
<!-- Polygraphy Test: XFAIL End -->

为了增加此命令因上述原因失败的可能性，
我们将强制 `Add` 以 FP16 精度运行，而后续的 `Sub` 以 FP32 运行。
这将阻止它们被融合，并导致 `Add` 的输出溢出 FP16 范围。


### 使用网络后处理脚本约束精度

另一个选项是使用 TensorRT 网络后处理脚本在解析的网络上应用精度。

使用提供的网络后处理脚本 [add_constraints.py](./add_constraints.py) 来约束模型中的精度：

```
polygraphy run needs_constraints.onnx --onnxrt --trt --fp16 --precision-constraints obey \
    --val-range x:[1,2] --check-error-stat median \
    --trt-network-postprocess-script ./add_constraints.py
```

*提示：您可以使用 `--trt-npps` 作为 `--trt-network-postprocess-script` 的简写。*

默认情况下，Polygraphy 在脚本中查找名为 `postprocess` 的函数来执行。要指定
要使用的不同函数，请在脚本名称后加上冒号和函数名称，例如

<!-- Polygraphy Test: Ignore Start -->
```
polygraphy run ... --trt-npps my_script.py:custom_func
```
<!-- Polygraphy Test: Ignore End -->


### 使用网络加载器脚本约束精度

或者，您可以使用网络加载器脚本手动定义整个网络，
作为其中的一部分，您可以设置层精度。

以下部分假设您已经阅读了关于
[手动定义 TensorRT 网络或配置](../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)
的示例，并且对如何使用 [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html) 有基本的了解。

首先，在模型上运行 ONNX-Runtime 以生成参考输入和黄金输出：

```bash
polygraphy run needs_constraints.onnx --onnxrt --val-range x:[1,2] \
    --save-inputs inputs.json --save-outputs golden_outputs.json
```

接下来，运行提供的网络加载器脚本
[constrained_network.py](./constrained_network.py)，该脚本约束
模型中的精度，强制 TensorRT 遵守约束，使用保存的输入并与保存的黄金输出进行比较：

```bash
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median
```

请注意，除了
显式约束的层之外，TensorRT 可能会选择在 FP32 中运行网络中的其他层，如果这样做会导致更高的整体
引擎性能。

**[可选]**：运行网络脚本，但允许 TensorRT 在必要时忽略精度
约束。如果 TensorRT
没有满足所请求精度约束的层实现，则可能需要这样做：

```
polygraphy run constrained_network.py --precision-constraints prefer \
    --trt --fp16 --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median
```


## 另请参阅

*   [使用降低的精度](../../../../how-to/work_with_reduced_precision.md) 有关如何使用 Polygraphy 调试
    降低精度优化的更通用指南。
*   [手动定义 TensorRT 网络或配置](../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually) 有关
    如何创建网络脚本模板的说明。
*   [TensorRT Python API 参考](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html)
