# TensorRT 中的 Int8 校准


## 简介

TensorRT 中的 Int8 校准涉及在引擎构建过程中向 TensorRT 提供一组有代表性的输入数据。TensorRT 中包含的[校准 API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/Calibrator.html)
要求用户处理将输入数据复制到 GPU 并管理 TensorRT 生成的校准缓存。

虽然 TensorRT API 提供了更高程度的控制，但我们可以为许多常见用例大大简化这一过程。为此，Polygraphy 提供了一个校准器，它可以与 Polygraphy 一起使用，也可以直接与 TensorRT 一起使用。在后一种情况下，Polygraphy 校准器的行为完全像普通的 TensorRT int8 校准器。

在这个示例中，我们将了解如何使用 Polygraphy 的校准器用（伪造的）校准数据校准网络，以及如何仅用单个参数就能管理校准缓存。


## 运行示例

1. 安装先决条件
    * 确保安装了 TensorRT
    * 使用 `python3 -m pip install -r requirements.txt` 安装其他依赖项

2. 运行示例：

    ```bash
    python3 example.py
    ```

3. 第一次运行示例时，它将创建一个名为 `identity-calib.cache` 的校准缓存。
    如果您再次运行示例，您应该看到它现在使用缓存而不是再次运行校准：

    ```bash
    python3 example.py
    ```


## 扩展阅读

有关 TensorRT 中 int8 校准工作原理的更多信息，请参阅
[开发人员指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c)
