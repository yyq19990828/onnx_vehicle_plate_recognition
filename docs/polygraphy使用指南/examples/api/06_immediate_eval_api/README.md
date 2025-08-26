# 立即评估的功能性 API

## 简介

<!-- Polygraphy Test: Ignore Start -->
大多数情况下，Polygraphy 附带的延迟加载器有几个优点：

- 它们允许我们将工作推迟到我们实际需要做的时候，这可能会节省时间。
- 由于构造的加载器非常轻量级，使用延迟评估加载器的运行器可以轻松地复制到其他进程或线程中，然后在那里启动它们。如果运行器引用的是整个模型/推理会话，那么以这种方式复制它们将并非易事。
- 它们允许我们通过将加载器链接在一起来预先定义一系列操作，这提供了一种构建可重用函数的简单方法。例如，我们可以创建一个从 ONNX 导入模型并生成序列化 TensorRT 引擎的加载器：

    ```python
    build_engine = EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))
    ```

- 它们允许特殊的语义，即如果向加载器提供可调用对象，它将获得返回值的所​​有权，否则则不会。这些特殊的语义对于在多个加载器之间共享对象很有用。

然而，这有时会导致代码可读性较差，甚至完全令人困惑。例如，考虑以下内容：
```python
# 这个例子中的每一行看起来几乎都一样，但行为却大相径庭。
# 其中一些行甚至会导致内存泄漏！
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")) # 这是一个加载器实例，而不是引擎！
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))() # 这是一个引擎。
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")()) # 又是一个加载器实例...
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")())() # 回到引擎！
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))()() # 这会抛出异常 - 你能看出为什么吗？
```

因此，Polygraphy 提供了每个加载器的立即评估的功能等价物。每个功能变体都使用与加载器相同的名称，但使用 `snake_case` 而不是 `PascalCase`。使用功能变体，像这样的加载器代码：

```python
parse_network = NetworkFromOnnxPath("/path/to/model.onnx")
create_config = CreateConfig(fp16=True, tf32=True)
build_engine = EngineFromNetwork(parse_network, create_config)
engine = build_engine()
```

变成：

```python
builder, network, parser = network_from_onnx_path("/path/to/model.onnx")
config = create_config(builder, network, fp16=True, tf32=True)
engine = engine_from_network((builder, network, parser), config)
```
<!-- Polygraphy Test: Ignore End -->


在此示例中，我们将了解如何利用功能性 API 将 ONNX 模型转换为 TensorRT 网络，修改网络，构建启用 FP16 精度的 TensorRT 引擎，并运行推理。我们还将引擎保存到文件中，以了解如何再次加载它并运行推理。


## 运行示例

1.  安装先决条件
    *   确保已安装 TensorRT
    *   使用 `python3 -m pip install -r requirements.txt` 安装其他依赖项

2.  **[可选]** 在运行示例前检查模型：

    ```bash
    polygraphy inspect model identity.onnx
    ```

3.  运行构建和运行引擎的脚本：

    ```bash
    python3 build_and_run.py
    ```

4.  **[可选]** 检查示例构建的 TensorRT 引擎：

    ```bash
    polygraphy inspect model identity.engine
    ```

5.  运行加载先前构建的引擎，然后运行它的脚本：

    ```bash
    python3 load_and_run.py
    ```
