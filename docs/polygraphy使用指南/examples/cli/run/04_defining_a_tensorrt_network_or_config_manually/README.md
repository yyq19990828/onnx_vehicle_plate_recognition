# 手动定义 TensorRT 网络或配置


## 简介

在某些情况下，使用 Python API 从头开始定义 TensorRT 网络，
或修改通过其他方式（例如解析器）创建的网络可能很有用。通常，这会限制您
使用 CLI 工具，至少在您构建引擎之前是这样，因为网络无法序列化
到磁盘并在命令行上加载。

Polygraphy CLI 工具为此提供了一个变通方法 - 如果您的 Python 脚本定义了一个
名为 `load_network` 的函数，该函数不带参数并返回一个 TensorRT 构建器、网络，
以及可选的解析器，那么您可以提供您的 Python 脚本来代替模型参数。

同样，我们可以使用一个脚本创建一个自定义的 TensorRT 构建器配置，该脚本定义
一个名为 `load_config` 的函数，该函数接受一个构建器和网络并返回一个构建器配置。

在本示例中，包含的 `define_network.py` 脚本解析一个 ONNX 模型并向其附加一个恒等
层。由于它在一个名为 `load_network` 的函数中返回构建器、网络和解析器，
我们可以仅使用一个命令从中构建并运行一个 TensorRT 引擎。`create_config.py`
脚本创建一个新的 TensorRT 构建器配置并启用 FP16 模式。


### 提示：自动生成脚本模板

您可以不从头开始编写网络脚本，而是使用
`polygraphy template trt-network` 为您提供一个起点：

```bash
polygraphy template trt-network -o my_define_network.py
```

如果您想从一个模型开始并修改生成的 TensorRT 网络，而不是
从头开始创建一个，只需将模型作为参数提供给 `template trt-network`：

```bash
polygraphy template trt-network identity.onnx -o my_define_network.py
```

同样，您可以使用 `polygraphy template trt-config` 为配置生成一个模板脚本：

```bash
polygraphy template trt-config -o my_create_config.py
```

您还可以指定构建器配置选项来预填充脚本。
例如，要启用 FP16 模式：

```bash
polygraphy template trt-config --fp16 -o my_create_config.py
```


## 运行示例

1.  运行 `define_network.py` 中定义的网络：

    ```bash
    polygraphy run --trt define_network.py --model-type=trt-network-script
    ```

2.  使用 `create_config.py` 中定义的构建器配置运行步骤 (1) 中的网络：

    ```bash
    polygraphy run --trt define_network.py --model-type=trt-network-script --trt-config-script=create_config.py
    ```

    请注意，我们可以在同一个脚本中定义 `load_network` 和 `load_config`。
    实际上，我们可以从任意脚本甚至模块中检索这些函数。

*提示：我们可以对 `polygraphy convert` 使用相同的方法来构建但不运行引擎。*
