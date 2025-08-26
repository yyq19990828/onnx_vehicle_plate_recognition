# 将 ONNX 模型转换为 FP16

## 简介

在使用 TensorRT 降低精度优化（`--fp16` 和 `--tf32` 标志）对 FP32 训练的 ONNX 模型进行调试精度问题时，
将模型转换为 FP16 并在 ONNX-Runtime 下运行它会很有帮助，
以检查是否存在以降低的精度运行模型所固有的问题。

## 运行示例

1.  将模型转换为 FP16：

    ```bash
    polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx
    ```

2.  **[可选]** 检查生成的模型：

    ```bash
    polygraphy inspect model identity_fp16.onnx
    ```

3.  **[可选]** 在 ONNX-Runtime 下运行 FP32 和 FP16 模型，然后比较结果：

    ```bash
    polygraphy run --onnxrt identity.onnx \
       --save-inputs inputs.json --save-outputs outputs_fp32.json
    ```

    ```bash
    polygraphy run --onnxrt identity_fp16.onnx \
       --load-inputs inputs.json --load-outputs outputs_fp32.json \
       --atol 0.001 --rtol 0.001
    ```

4.  **[可选]** 检查 FP16 模型的任何中间输出是否
    包含 NaN 或无穷大（请参阅 [检查中间 NaN 或无穷大](../../../../examples/cli/run/07_checking_nan_inf)）：

    ```bash
    polygraphy run --onnxrt identity_fp16.onnx --onnx-outputs mark all --validate
    ```

## 另请参阅

*   [跨运行比较](../../../../examples/cli/run/02_comparing_across_runs)
*   [检查中间 NaN 或无穷大](../../../../examples/cli/run/07_checking_nan_inf)
*   [调试 TensorRT 精度问题](../../../../how-to/debug_accuracy.md)
