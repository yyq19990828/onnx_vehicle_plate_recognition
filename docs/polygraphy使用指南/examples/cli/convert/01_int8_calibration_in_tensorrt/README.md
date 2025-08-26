# TensorRT 中的 Int8 校准


## 简介

在 [API 示例 04](../../../api/04_int8_calibration_in_tensorrt/) 中，我们了解了如何利用
Polygraphy 附带的校准器轻松地使用 TensorRT 运行 int8 校准。

但是，如果我们想在命令行上做同样的事情呢？

为此，我们需要一种向命令行工具提供自定义输入数据的方法。
Polygraphy 提供了多种方法，详见[此处](../../../../how-to/use_custom_input_data.md)。

在此示例中，我们将使用一个数据加载器脚本，方法是在一个名为 `data_loader.py` 的 Python
脚本中定义一个 `load_data` 函数，然后使用 `polygraphy convert` 来构建 TensorRT 引擎。

*提示：我们可以使用类似的方法，使用 `polygraphy run` 来构建和运行引擎。*

## 运行示例

1.  转换模型，使用自定义数据加载器脚本提供校准数据，
    并保存校准缓存以备将来使用：

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache identity_calib.cache \
        -o identity.engine
    ```

2.  **[可选]** 使用缓存重建引擎以跳过校准：

    ```bash
    polygraphy convert identity.onnx --int8 \
        --calibration-cache identity_calib.cache \
        -o identity.engine
    ```

    由于校准缓存已经填充，校准将被跳过。
    因此，我们*不*需要提供输入数据。


3.  **[可选]** 直接从 API 示例中使用数据加载器。

    这里概述的方法非常灵活，我们甚至可以使用我们在 API 示例中定义的数据加载器！
    我们只需要指定函数名称，因为该示例没有将其命名为 `load_data`：

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ../../../api/04_int8_calibration_in_tensorrt/example.py:calib_data \
        -o identity.engine
    ```
