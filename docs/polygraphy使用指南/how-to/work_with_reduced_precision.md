# 使用降低的精度

## 模型能否以降低的精度运行？

TensorRT 通常可用于运行以 FP32 训练的模型，使用较低精度的实现（特别是 TF32 和 FP16），几乎不需要额外的努力。请注意，在使用 INT8 时通常情况并非如此，这需要额外的步骤，如[此处](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)所述，以达到可接受的精度。

您可以通过使用 Polygraphy 将生成的引擎与 ONNX-Runtime 进行比较，轻松检查生成的引擎是否满足精度要求。有关如何执行此操作的更详细说明，请参阅[比较框架](../examples/cli/run/01_comparing_frameworks)。

### 对 FP16 限制进行健全性检查

如果您正在使用 `--trt --fp16` 并且精度不可接受，您可以通过使用 ONNX-Runtime 在 FP16 中运行相同的模型来健全性检查这是否是使用降低精度模型的限制（而不是 TensorRT 特定的问题）。有关生成用于比较的模型并验证其输出的说明，请参阅[将 ONNX 模型转换为 FP16](../examples/cli/convert/04_converting_models_to_fp16)。如果在 ONNX-Runtime 中以 FP16 运行模型失败，那么您可能需要调整模型或添加精度约束，如下所述，以达到可接受的精度。

## 调试精度失败

如果输出比较失败，下一步通常是隔离模型中导致精度失败的有问题的层。有关如何执行此操作的技术，请参阅[如何调试精度](../how-to/debug_accuracy.md)。

## 覆盖精度约束

一旦您确定了有问题的层，下一步就是将这些层的精度约束覆盖为 FP32，以查看精度是否恢复。有关如何使用 Polygraphy 试验不同精度约束的详细信息，请参阅[覆盖精度约束](../examples/cli/run/08_adding_precision_constraints)的示例。

## 其他选项

如果将层回退到 FP32 不足以恢复精度或导致不希望的性能下降，您通常需要修改和重新训练模型，以帮助将中间激活的动态范围保持在可表达的范围内。一些有助于实现此目的的技术包括：

* 在训练模型时对输入值进行归一化（例如，将 RGB 输入数据缩放到 `[0, 1]`），并使用训练好的模型进行推理。
* 在训练模型时使用批量归一化和其他正则化技术。
* 在使用 INT8 时，请考虑使用[量化感知训练 (QAT)](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/) 来帮助提高精度。
