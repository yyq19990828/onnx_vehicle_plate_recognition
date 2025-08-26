# 检查 TensorRT ONNX 支持

## 简介

`inspect capability` 子工具提供有关 TensorRT 对给定 ONNX 图的 ONNX 算子支持的详细信息。
它还能够从原始模型中分区并保存支持和不支持的子图，以便报告给定模型的所有动态检查错误（请参阅示例 `08_inspecting_tensorrt_onnx_support`）。

## 运行示例

1.  生成能力报告

    ```bash
    polygraphy inspect capability nested_local_function.onnx
    ```

2.  这将显示一个摘要表，如下所示：

    ```
    [I] ===== 摘要 =====
        堆栈跟踪                                                                                | 算子      | 节点               | 原因
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        onnx_graphsurgeon_node_1 (OuterFunction) -> onnx_graphsurgeon_node_1 (NestedLocalFake2) | Fake_2    | nested_node_fake_2 | 在名称为 nested_node_fake_2、算子为 Fake_2 的节点 0 中 (checkFallbackPluginImporter): INVALID_NODE: creator && "找不到插件，插件名称、版本和命名空间是否正确？"
        onnx_graphsurgeon_node_1 (OuterFunction)                                                | Fake_1    | nested_node_fake_1 | 在名称为 nested_node_fake_1、算子为 Fake_1 的节点 0 中 (checkFallbackPluginImporter): INVALID_NODE: creator && "找不到插件，插件名称、版本和命名空间是否正确？"
    ```

## 理解输出

在此示例中，`nested_local_function.onnx` 包含 TensorRT 不支持的 `Fake_1` 和 `Fake_2` 节点。`Fake_1` 节点位于本地函数 `OuterFunction` 内，`Fake_2` 节点位于嵌套的本地函数 `NestedLocalFake2` 内。
摘要表显示了由本地函数组成的当前堆栈跟踪、发生错误的算子以及不支持的原因。

有关更多信息和选项，请参阅 `polygraphy inspect capability --help`。
