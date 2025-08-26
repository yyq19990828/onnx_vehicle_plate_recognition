# 使用 TensorRT 网络 API


## 简介

除了加载现有模型外，TensorRT 还允许您使用网络 API 手动定义网络。

在此示例中，我们将了解如何使用 Polygraphy 的 `extend` 装饰器（在[示例 03](../03_interoperating_with_tensorrt) 中介绍）与 `CreateNetwork` 加载器结合使用，以将使用 TensorRT API 定义的网络与 Polygraphy 无缝集成。


## 运行示例

1.  安装先决条件
    *   确保已安装 TensorRT
    *   使用 `python3 -m pip install -r requirements.txt` 安装其他依赖项

2.  **[可选]** 检查 `create_network()` 生成的 TensorRT 网络。
    这将从脚本内部调用 `create_network()` 并显示生成的 TensorRT 网络：

    ```bash
    polygraphy inspect model example.py --trt-network-func create_network --show layers attrs weights
    ```

3.  运行示例：

    ```bash
    python3 example.py
    ```

## 更多阅读

有关 TensorRT 网络 API 的更多信息，请参阅
[TensorRT API 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Graph/Network.html)
