# 检查中间的 NaN 或无穷大

## 简介

在 Polygraphy 中调试模型精度问题时，检查逐层输出以发现潜在问题会很有帮助。Polygraphy 的 `run` 子工具提供了一个有用的标志 `--validate`，可以快速诊断有问题的中间输出。

本示例演示了如何将此标志与一个通过向输入张量添加无穷大来有意生成无穷大输出的模型一起使用。

## 运行示例

 <!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy run add_infinity.onnx --onnx-outputs mark all --onnxrt --validate
```
 <!-- Polygraphy Test: XFAIL End -->

 <!-- Polygraphy Test: Ignore Start -->
您应该会看到如下输出：
```
[I] onnxrt-runner-N0-05/13/22-22:35:48  | 在 0.1326 毫秒内完成 1 次迭代 | 平均推理时间：0.1326 毫秒。
[I] 输出验证 | 运行器：['onnxrt-runner-N0-05/13/22-22:35:48']
[I]     onnxrt-runner-N0-05/13/22-22:35:48  | 验证输出：B (check_inf=True, check_nan=True)
[I]         mean=inf, std-dev=nan, var=nan, median=inf, min=inf at (0,), max=inf at (0,), avg-magnitude=inf
[E]         检测到无穷大 | 在此输出中遇到一个或多个非有限值
[I]         注意：使用 -vv 或将日志记录详细程度设置为 EXTRA_VERBOSE 以显示非有限值
[E]         失败 | 在输出中检测到错误：B
[E]     失败 | 输出验证
```
 <!-- Polygraphy Test: Ignore End -->

## 另请参阅

*   [调试 TensorRT 精度问题](../../../../how-to/debug_accuracy.md)
