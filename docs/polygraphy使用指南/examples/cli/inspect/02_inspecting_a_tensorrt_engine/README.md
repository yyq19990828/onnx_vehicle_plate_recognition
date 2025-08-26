# 检查 TensorRT 引擎


## 简介

`inspect model` 子工具可以加载并显示有关 TensorRT 引擎（即 plan 文件）的信息：


## 运行示例

1.  生成具有动态形状和 2 个配置文件的引擎：

    ```bash
    polygraphy run dynamic_identity.onnx --trt \
        --trt-min-shapes X:[1,2,1,1] --trt-opt-shapes X:[1,2,3,3] --trt-max-shapes X:[1,2,5,5] \
        --trt-min-shapes X:[1,2,2,2] --trt-opt-shapes X:[1,2,4,4] --trt-max-shapes X:[1,2,6,6] \
        --save-engine dynamic_identity.engine
    ```

2.  检查引擎：

    ```bash
    polygraphy inspect model dynamic_identity.engine \
        --show layers
    ```

    注意：`--show layers` 仅在引擎构建时 `profiling_verbosity` 不为 `NONE` 时才有效。
        更高的详细程度可以提供更多逐层信息。

    这将显示如下内容：

    ```
    [I] ==== TensorRT 引擎 ====
        名称: Unnamed Network 0 | Explicit Batch Engine

        ---- 1 个引擎输入 ----
        {X [dtype=float32, shape=(1, 2, -1, -1)]}

        ---- 1 个引擎输出 ----
        {Y [dtype=float32, shape=(1, 2, -1, -1)]}

        ---- 内存 ----
        设备内存: 0 字节

        ---- 2 个配置文件（每个 2 个张量）----
        - 配置文件: 0
            张量: X          (输入), 索引: 0 | 形状: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
            张量: Y         (输出), 索引: 1 | 形状: (1, 2, -1, -1)

        - 配置文件: 1
            张量: X          (输入), 索引: 0 | 形状: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
            张量: Y         (输出), 索引: 1 | 形状: (1, 2, -1, -1)

        ---- 每个配置文件 1 个层 ----
        - 配置文件: 0
            层 0    | node_of_Y [Op: Reformat]
                {X [shape=(1, 2, -1, -1)]}
                 -> {Y [shape=(1, 2, -1, -1)]}

        - 配置文件: 1
            层 0    | node_of_Y [profile 1] [Op: MyelinReformat]
                {X [profile 1] [shape=(1, 2, -1, -1)]}
                 -> {Y [profile 1] [shape=(1, 2, -1, -1)]}
    ```

    也可以使用 `--show layers attrs` 显示更详细的层信息。
