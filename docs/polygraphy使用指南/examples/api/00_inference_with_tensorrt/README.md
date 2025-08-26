# 转换为 TensorRT 并运行推理


## 简介

Polygraphy 包含一个高级 Python API，可以转换模型并使用各种后端运行推理。有关 Polygraphy
Python API 的概述，请参阅[此处](../../../polygraphy/)。

在此示例中，我们将了解如何利用该 API 轻松地将 ONNX 模型转换为 TensorRT，并在启用 FP16 精度的情况下运行推理。然后，我们会将引擎保存到文件中，并了解如何再次加载它并运行推理。


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

## 更多阅读

有关 Polygraphy Python API 的更多详细信息，请参阅
[Polygraphy API 参考](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/polygraphy/index.html)。
