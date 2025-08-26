# 检查 ONNX 模型


## 简介

`inspect model` 子工具可以显示 ONNX 模型。


## 运行示例

1.  检查 ONNX 模型：

    ```bash
    polygraphy inspect model identity.onnx --show layers
    ```

    这将显示如下内容：

    ```
    [I] ==== ONNX 模型 ====
        名称: test_identity | ONNX Opset: 8

        ---- 1 个图形输入 ----
        {x [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 1 个图形输出 ----
        {y [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 0 个初始化器 ----
        {}

        ---- 1 个节点 ----
        节点 0    |  [Op: Identity]
            {x [dtype=float32, shape=(1, 1, 2, 2)]}
             -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
    ```

    也可以使用 `--show layers attrs weights` 显示详细的层信息，包括层属性。
