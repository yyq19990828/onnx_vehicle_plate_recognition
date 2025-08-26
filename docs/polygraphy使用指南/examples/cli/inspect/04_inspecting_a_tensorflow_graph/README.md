# 检查 TensorFlow 图


## 简介

`inspect model` 子工具可以显示 TensorFlow 图。


## 运行示例

1.  检查 TensorFlow 冻结模型：

    ```bash
    polygraphy inspect model identity.pb --model-type=frozen
    ```

    这将显示如下内容：

    ```
    [I] ==== TensorFlow 图 ====
        ---- 1 个图输入 ----
        {Input:0 [dtype=float32, shape=(1, 15, 25, 30)]}

        ---- 1 个图输出 ----
        {Identity_2:0 [dtype=float32, shape=(1, 15, 25, 30)]}

        ---- 4 个节点 ----
    ```
