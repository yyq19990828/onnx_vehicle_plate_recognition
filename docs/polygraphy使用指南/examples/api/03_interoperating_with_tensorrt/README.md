# 与 TensorRT 互操作


## 简介

Polygraphy 的一个关键特性是与 TensorRT 以及其他后端的完全互操作性。由于 Polygraphy 不会隐藏底层的后端 API，因此可以自由地在使用 Polygraphy API 和后端 API（例如 TensorRT）之间切换。

在此示例中，我们将了解如何在不放弃 Polygraphy 提供的便利的情况下，保留对后端提供的高级功能的访问权限——两全其美。

Polygraphy 提供了一个 `extend` 装饰器，可用于轻松扩展现有的 Polygraphy 加载器。这在许多场景中都很有用，但在此示例中，我们将重点关注您可能希望：
- 在构建引擎之前修改 TensorRT 网络
- 使用 Polygraphy 当前不支持的 TensorRT 构建器标志


## 运行示例

1.  安装先决条件
    *   确保已安装 TensorRT
    *   使用 `python3 -m pip install -r requirements.txt` 安装其他依赖项


2.  **[可选]** 检查 `load_network()` 生成的 TensorRT 网络。
    这将从脚本内部调用 `load_network()` 并显示生成的 TensorRT 网络，该网络应命名为 `"MyIdentity"`：

    ```bash
    polygraphy inspect model example.py --trt-network-func load_network --show layers attrs weights
    ```

3.  运行示例：

    ```bash
    python3 example.py
    ```
