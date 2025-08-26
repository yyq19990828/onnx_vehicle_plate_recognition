# 使用 PyTorch 张量

## 简介

一些运行器如 `OnnxrtRunner` 和 `TrtRunner` 除了支持 NumPy 数组外，还可以接受和返回 PyTorch 张量。当在输入中提供 PyTorch 张量时，运行器也会将输出作为 PyTorch 张量返回。这在 PyTorch 支持 NumPy 不支持的数据类型（如 BFloat16）的情况下特别有用。

Polygraphy 包含的 TensorRT `Calibrator` 也可以直接接受 PyTorch 张量。

此示例在可能的情况下（即如果安装了支持 GPU 的 PyTorch 版本）在 GPU 上使用 PyTorch 张量。当张量已经驻留在 GPU 内存中时，运行器/校准器中不需要额外的复制。

## 运行示例

1. 安装先决条件
    * 确保安装了 TensorRT
    * 使用 `python3 -m pip install -r requirements.txt` 安装其他依赖项


2. 运行示例：

    ```bash
    python3 example.py
    ```


## 另请参阅

* [使用 TensorRT 进行推理](../00_inference_with_tensorrt/)
* [TensorRT 中的 INT8 校准](../04_int8_calibration_in_tensorrt/)
