# 检查策略重放文件


## 简介

`inspect tactics` 子工具可以显示有关由 Polygraphy 生成的 TensorRT 策略重放文件的信息。


## 运行示例

1.  生成一个策略重放文件：

    ```bash
    polygraphy run model.onnx --trt --save-tactics replay.json
    ```

2.  检查策略重放：

    ```bash
    polygraphy inspect tactics replay.json
    ```

    这将显示如下内容：

    ```
    [I] 层: ONNXTRT_Broadcast
            算法: (实现: 2147483661, 策略: 0) | 输入: (('DataType.FLOAT'),) | 输出: (('DataType.FLOAT'),)
        层: node_of_z
            算法: (实现: 2147483651, 策略: 1) | 输入: (('DataType.FLOAT'), ('DataType.FLOAT')) | 输出: (('DataType.FLOAT'),)
    ```
