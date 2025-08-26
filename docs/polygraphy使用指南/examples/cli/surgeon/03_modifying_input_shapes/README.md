# 修改输入形状

## 简介

`surgeon sanitize` 子工具可用于修改 ONNX 模型的输入形状。
这不会改变模型的中间层，因此，如果模型对输入形状做出假设（例如，具有硬编码新形状的 `Reshape` 节点），则可能会导致问题。

输出形状可以被推断出来，因此它们不会被修改（也不需要修改）。

*注意：强烈建议使用所需的形状重新导出 ONNX 模型。*
    *此处显示的方法仅应在无法执行此操作时使用。*

## 运行示例

1.  将模型的输入形状更改为具有动态批量维度的形状，同时保持其他维度不变：

    ```bash
    polygraphy surgeon sanitize identity.onnx \
        --override-input-shapes x:['batch',1,2,2] \
        -o dynamic_identity.onnx
    ```

2.  **[可选]** 您可以使用 `inspect model` 来确认它是否看起来正确：

    ```bash
    polygraphy inspect model dynamic_identity.onnx --show layers
    ```
