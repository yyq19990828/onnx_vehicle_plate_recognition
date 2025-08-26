# 调试不稳定的 TensorRT 策略

**重要提示：此示例在较新版本的 TensorRT 中不再可靠地工作，因为它们做出了一些**
    **未通过 IAlgorithmSelector 接口公开的策略选择（在 TensorRT 10.8 中已弃用。**
    **请改用 ITimingCache 中的可编辑模式）。因此，下面概述的方法**
    **无法保证确定性的引擎构建。对于 TensorRT 8.7 及更高版本，您可以使用**
    **策略计时缓存（Polygraphy 中的 `--save-timing-cache` 和 `--load-timing-cache`）来确保**
    **确定性，但这些文件是不透明的，因此无法由 `inspect diff-tactics` 解释**

## 简介

有时，TensorRT 中的一个策略可能会产生不正确的结果，或者有
其他错误的行为。由于 TensorRT 构建器依赖于计时
策略，引擎构建是非确定性的，这可能使策略错误
表现为不稳定/间歇性的失败。

解决这个问题的一种方法是多次运行构建器，
从每次运行中保存策略重放文件。一旦我们有了一组已知的良好和
已知的错误策略，我们就可以比较它们以确定哪个策略
可能是错误的来源。

`debug build` 子工具允许您自动化此过程。

有关 `debug` 工具如何工作的更多详细信息，请参阅帮助输出：
`polygraphy debug -h` 和 `polygraphy debug build -h`。


## 运行示例

1.  从 ONNX-Runtime 生成黄金输出：

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-outputs golden.json
    ```

2.  使用 `debug build` 重复构建 TensorRT 引擎并根据黄金输出比较结果，
    每次保存一个策略重放文件：

    ```bash
    polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
        --artifacts-dir replays --artifacts replay.json --until=10 \
        --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
    ```

    让我们分解一下：

    -   像其他 `debug` 子工具一样，`debug build` 在每次迭代中生成一个中间产物
        （默认为 `./polygraphy_debug.engine`）。在这种情况下，这个产物是一个 TensorRT 引擎。

        *提示：`debug build` 支持所有其他工具（如 `convert` 或 `run`）支持的 TensorRT 构建器配置选项。*

    -   为了让 `debug build` 确定每个引擎是失败还是通过，
        我们提供一个 `--check` 命令。由于我们正在查看一个（假的）精度问题，
        我们可以使用 `polygraphy run` 来比较引擎的输出与我们的黄金值。

        *提示：像其他 `debug` 子工具一样，也支持交互模式，您只需*
            *省略 `--check` 参数即可使用。*

    -   与其他 `debug` 子工具不同，`debug build` 没有自动终止条件，所以我们需要
        提供 `--until` 选项，以便工具知道何时停止。这可以是一个
        迭代次数，也可以是 `"good"` 或 `"bad"`。在后一种情况下，工具将在找到
        第一个通过或失败的迭代后分别停止。

    -   由于我们最终想要比较好的和坏的策略重放，我们指定 `--save-tactics`
        来保存每次迭代的策略重放文件，然后使用 `--artifacts` 来告诉 `debug build`
        管理它们，这包括将它们排序到
        主产物目录下的 `good` 和 `bad` 子目录中，该目录由 `--artifacts-dir` 指定。


3.  使用 `inspect diff-tactics` 来确定哪些策略可能是坏的：

    ```bash
    polygraphy inspect diff-tactics --dir replays
    ```

    *注意：最后一步应该报告它无法确定潜在的坏策略，因为*
        *我们的 `bad` 目录此时应该是空的（否则请提交 TensorRT 问题！）：*

    <!-- Polygraphy Test: Ignore Start -->
    ```
    [I] Loaded 2 good tactic replays.
    [I] Loaded 0 bad tactic replays.
    [I] Could not determine potentially bad tactics. Try generating more tactic replay files?
    ```
    <!-- Polygraphy Test: Ignore End -->


## 进一步阅读

有关 `debug` 工具的更多信息，以及适用于
所有 `debug` 子工具的提示和技巧，请参阅
[`debug` 子工具的使用指南](../../../../how-to/use_debug_subtools_effectively.md)。
