# 检查推理输出


## 简介

`inspect data` 子工具可以显示有关 `Comparator.run()` 生成的 `RunResults` 对象的信息，该对象表示推理输出。


## 运行示例

1.  使用 ONNX-Runtime 生成一些推理输出：

    ```bash
    polygraphy run identity.onnx --onnxrt --save-outputs outputs.json
    ```

2.  检查结果：

    ```bash
    polygraphy inspect data outputs.json --show-values
    ```

    这将显示如下内容：

    ```
    [I] ==== 运行结果 (1 个运行器) ====

        ---- onnxrt-runner-N0-07/15/21-10:46:07 (1 次迭代) ----

        y [dtype=float32, shape=(1, 1, 2, 2)] | 统计: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995, p90=0.62933, p95=0.67483, p99=0.71123
            [[[[4.17021990e-01 7.20324516e-01]
               [1.14374816e-04 3.02332580e-01]]]]
    ```
