# 检查 TensorRT ONNX 支持

## 简介

`inspect capability` 子工具提供有关 TensorRT 对给定 ONNX 图的 ONNX 算子支持的详细信息。
它还能够从原始模型中分区并保存支持和不支持的子图，以便报告给定模型的所有动态检查错误。

## 运行示例

1.  生成能力报告

    ```bash
    polygraphy inspect capability --with-partitioning model.onnx
    ```

2.  这将显示一个摘要表，如下所示：

    ```
    [I] ===== 摘要 =====
        算子   | 计数    | 原因                                                                                                                                                                      | 节点
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Fake     |       1 | 在名称为：、算子为：Fake 的节点 0 中 (checkFallbackPluginImporter): INVALID_NODE: creator && "找不到插件，插件名称、版本和命名空间是否正确？" | [[2, 3]]
    ```

## 理解输出

在此示例中，`model.onnx` 包含一个 TensorRT 不支持的 `Fake` 节点。
摘要表显示了不支持的算子、不支持的原因、它在图中出现的次数，
以及这些节点在图中的索引范围，以防连续有多个不支持的节点。
请注意，此范围使用包含起始索引和不包含结束索引。

需要注意的是，图分区逻辑 (`--with-partitioning`) 目前不支持显示本地函数 (`FunctionProto`) 内节点的问题。有关正确处理本地函数内节点的静态错误报告，请参阅默认流程的描述（不带 `--with-partitioning` 选项，在示例 `09_inspecting_tensorrt_static_onnx_support` 中描述）。

有关更多信息和选项，请参阅 `polygraphy inspect capability --help`。
