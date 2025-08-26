# 在 TensorRT 中处理具有动态形状的模型

## 简介

为了在 TensorRT 中使用动态输入形状，我们必须在构建引擎时指定一个或多个可能的形状范围。
有关其工作原理的详细信息，请参阅
[API 示例 07](../../../api/07_tensorrt_and_dynamic_shapes/)。

使用 CLI 时，我们可以为每个输入指定一次或多次最小、最优和最大形状。如果每个输入指定了多次形状，则会创建多个优化配置文件。

## 运行示例

1.  构建具有 3 个独立配置文件的引擎：

    ```bash
    polygraphy convert dynamic_identity.onnx -o dynamic_identity.engine \
        --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[1,3,28,28] --trt-max-shapes X:[1,3,28,28] \
        --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[4,3,28,28] --trt-max-shapes X:[32,3,28,28] \
        --trt-min-shapes X:[128,3,28,28] --trt-opt-shapes X:[128,3,28,28] --trt-max-shapes X:[128,3,28,28]
    ```

    对于具有多个输入的模型，只需为每个 `--trt-*-shapes` 参数提供多个参数。
    例如：`--trt-min-shapes input0:[10,10] input1:[10,10] input2:[10,10] ...`

    *提示：如果我们只想使用一个 min == opt == max 的配置文件，我们可以利用运行时输入*
        *形状选项：`--input-shapes` 作为一种方便的简写，而不是分别设置 min/opt/max。*


2.  **[可选]** 检查生成的引擎：

    ```bash
    polygraphy inspect model dynamic_identity.engine
    ```


## 进一步阅读

有关在 TensorRT 中使用动态形状的更多信息，请参阅
[开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)
