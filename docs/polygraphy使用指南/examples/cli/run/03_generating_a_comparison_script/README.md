
# 生成用于高级比较的脚本


## 简介

对于更高级的需求，您可能需要使用 [API](../../../../polygraphy)。
您可以使用 `run` 的 `--gen-script` 选项来创建一个 Python 脚本，而不是从头开始编写脚本，
您可以将其用作起点。


## 运行示例

1.  生成一个比较脚本：

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --gen-script=compare_trt_onnxrt.py
    ```

    生成的脚本将执行与 `run` 命令完全相同的操作。

2.  运行比较脚本，可以选择在修改后运行：

    ```bash
    python3 compare_trt_onnxrt.py
    ```
