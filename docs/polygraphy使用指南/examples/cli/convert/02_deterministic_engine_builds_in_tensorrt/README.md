# TensorRT 中的确定性引擎构建

**注意：此示例需要 TensorRT 8.7 或更高版本。**

## 简介

在引擎构建期间，TensorRT 运行并计时多个内核，以选择最优化的内核。由于内核计时可能因运行而异，因此该过程本质上是不确定的。

在许多情况下，确定性引擎构建可能是可取的。实现此目的的一种方法是使用计时缓存来确保每次都选择相同的内核。

## 运行示例

1.  构建引擎并保存计时缓存：

    ```bash
    polygraphy convert identity.onnx \
        --save-timing-cache timing.cache \
        -o 0.engine
    ```

2.  将计时缓存用于另一个引擎构建：

    ```bash
    polygraphy convert identity.onnx \
        --load-timing-cache timing.cache --error-on-timing-cache-miss \
        -o 1.engine
    ```

    我们指定 `--error-on-timing-cache-miss` 以便我们可以确保新引擎对每个层都使用了计时缓存中的条目。

3.  验证引擎完全相同：

    <!-- Polygraphy Test: Ignore Start -->
    ```bash
    diff <(polygraphy inspect model 0.engine --show layers attrs) <(polygraphy inspect model 1.engine --show layers attrs)
    ```
    <!-- Polygraphy Test: Ignore End -->
