# 检查 TensorRT 网络


## 简介

`inspect model` 子工具可以自动将支持的格式转换为 TensorRT 网络，然后显示它们。


## 运行示例

1.  解析 ONNX 模型后显示 TensorRT 网络：

    ```bash
    polygraphy inspect model identity.onnx \
        --show layers --display-as=trt
    ```

    这将显示如下内容：

    ```
    [I] ==== TensorRT 网络 ====
        名称: Unnamed Network 0 | Explicit Batch Network

        ---- 1 个网络输入 ----
        {x [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 1 个网络输出 ----
        {y [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 1 个层 ----
        Layer 0    | node_of_y [Op: LayerType.IDENTITY]
            {x [dtype=float32, shape=(1, 1, 2, 2)]}
             -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
    ```

    也可以使用 `--show layers attrs weights` 显示详细的层信息，包括层属性。
