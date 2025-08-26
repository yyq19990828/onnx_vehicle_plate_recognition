# 使用 Sanitize 为无界数据依赖形状 (DDS) 设置上限

## 简介

`surgeon sanitize` 子工具可用于为无界数据依赖形状 (DDS) 设置上限。
当一个张量的形状取决于另一个张量的运行时值时，这种形状称为 DDS。
一些 DDS 具有有限的上限。例如，`NonZero` 算子的输出形状是 DDS，但其输出形状不会超过其输入的形状。
然而，其他一些 DDS 没有上限。例如，当 `limit` 输入是运行时张量时，`Range` 算子的输出具有无界的 DDS。
具有无界 DDS 的张量对于 TensorRT 在构建阶段优化推理性能和内存使用非常困难。
在最坏的情况下，它们可能导致 TensorRT 引擎构建失败。

在本例中，我们将使用 polygraphy 为图中的无界 DDS 设置上限：

![./model.png](./model.png)

## 运行示例

1.  首先对模型运行常量折叠：

    ```bash
    polygraphy surgeon sanitize model.onnx -o folded.onnx --fold-constants
    ```

    请注意，列出无界 DDS 和设置上限需要常量折叠和符号形状推断。

2.  使用以下命令查找具有无界 DDS 的张量：

    ```bash
    polygraphy inspect model folded.onnx --list-unbounded-dds
    ```

    Polygraphy 将显示所有具有无界 DDS 的张量。

3.  使用以下命令为无界 DDS 设置上限：

    ```bash
    polygraphy surgeon sanitize folded.onnx --set-unbounded-dds-upper-bound 1000 -o modified.onnx 
    ```

    Polygraphy 将首先搜索所有具有无界 DDS 的张量。
    然后它将插入具有提供的上限值的 min 算子来限制 DDS 张量的大小。
    在本例中，在 `Range` 算子之前插入了一个 min 算子。
    使用修改后的模型，TensorRT 将知道 `Range` 算子的输出形状不会超过 1000。
    因此，可以为后续层选择更多的内核。

    ![./modified.png](./modified.png)

4.  检查现在是否没有具有无界 DDS 的张量：

    ```bash
    polygraphy inspect model modified.onnx --list-unbounded-dds
    ```

    现在 `modified.onnx` 应该不包含无界 DDS。

