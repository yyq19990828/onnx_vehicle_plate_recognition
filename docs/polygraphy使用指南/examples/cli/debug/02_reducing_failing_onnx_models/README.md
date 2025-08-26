# 减少失败的 ONNX 模型

## 简介

当一个模型由于任何原因失败时（例如，TensorRT 中的精度问题），
将其减少到触发失败的最小可能子图通常很有用。这使得
更容易查明失败的原因。

一种方法是生成原始 ONNX 模型的连续较小的子图。
在每次迭代中，我们可以检查子图是工作还是仍然失败；一旦我们有了一个工作的
子图，我们就知道上一次迭代生成的子图是最小的失败
子图。

`debug reduce` 子工具允许我们自动化这个过程。


## 运行示例

在本示例中，我们假设我们的模型 (`./model.onnx`) 在 TensorRT 中存在精度问题。
由于该模型实际上在 TensorRT 中可以工作（如果不行，请报告错误！），
我们将概述您通常会运行的命令，然后是您可以运行以
模拟失败的命令，以便了解该工具在实践中的外观。

我们的模拟失败将在模型中存在 `Mul` 节点时触发：

![./model.png](./model.png)

因此，最终简化的模型应该只包含 `Mul` 节点（因为其他节点不会导致失败）。

1.  对于使用动态输入形状或包含形状操作的模型，冻结输入
    形状并使用以下命令折叠形状操作：

    ```bash
    polygraphy surgeon sanitize model.onnx -o folded.onnx --fold-constants \
        --override-input-shapes x0:[1,3,224,224] x1:[1,3,224,224]
    ```

2.  我们假设 ONNX-Runtime 为我们提供了正确的输出。我们将从为网络中的每个张量生成黄金
    值开始。我们还将保存我们使用的输入：

    ```bash
    polygraphy run folded.onnx --onnxrt \
        --save-inputs inputs.json \
        --onnx-outputs mark all --save-outputs layerwise_golden.json
    ```

    然后，我们将使用 `data to-input` 子工具将输入和分层输出组合成一个单一的分层输入文件
    （我们将在下一步中看到为什么这是必要的）：

    ```bash
    polygraphy data to-input inputs.json layerwise_golden.json -o layerwise_inputs.json
    ```


3.  接下来，我们将在 `bisect` 模式下使用 `debug reduce`：

    ```bash
    polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect --load-inputs layerwise_inputs.json \
        --check polygraphy run polygraphy_debug.onnx --trt \
                --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json
    ```

    让我们分解一下：

    -   像其他 `debug` 子工具一样，`debug reduce` 在每次迭代中生成一个中间产物
        （默认为 `./polygraphy_debug.onnx`）。在这种情况下，这个产物是原始 ONNX 模型的某个子图。

    -   为了让 `debug reduce` 确定每个子图是失败还是通过，
        我们提供一个 `--check` 命令。由于我们正在调查一个精度问题，
        我们可以使用 `polygraphy run` 来与我们之前的黄金输出进行比较。

        *提示：像其他 `debug` 子工具一样，也支持交互模式，您只需*
            *省略 `--check` 参数即可使用。*

    -   在 `--check` 命令中，我们通过 `--load-inputs` 提供分层输入，因为否则，`polygraphy run`
        会为子图张量生成新的输入，这可能与我们生成黄金数据时这些张量的值不匹配。
        另一种方法是在 `debug reduce` 的每次迭代中运行参考实现
        （这里是 ONNX-Runtime），而不是提前运行。

    -   由于我们使用的是非默认输入数据，我们还直接向
        `debug reduce` 命令提供分层输入（除了提供给 `--check` 命令之外）。
        这在具有多个并行分支（*指模型中的路径而不是控制流*）的模型中很重要，例如：
        <!-- Polygraphy Test: Ignore Start -->
        ```
         inp0  inp1
          |     |
         Abs   Abs
            \ /
            Sum
             |
            out
        ```
        在这种情况下，`debug reduce` 需要能够用一个常量替换一个分支。
        为此，它需要知道您正在使用的输入数据，以便可以用正确的值替换它。
        虽然我们在这里使用一个文件，但输入数据可以通过
        [CLI 用户指南](../../../../how-to/use_custom_input_data.md)中介绍的任何其他 Polygraphy 数据加载器参数提供。

        如果您不确定是否需要提供数据加载器，
        `debug reduce` 在尝试替换分支时会发出如下警告：
        ```
        [W]     此模型包含多个分支/路径。为了继续简化，需要折叠一个分支。
                如果您的 `--check` 命令使用的是非默认数据加载器，请确保您已向 `debug reduce` 提供了数据加载器参数。
                否则可能会导致假阴性！
        ```
        <!-- Polygraphy Test: Ignore End -->

    -   我们指定 `-o` 选项，以便将简化的模型写入 `initial_reduced.onnx`。

    **模拟失败：** 我们可以将 `polygraphy inspect model` 与 `--fail-regex` 结合使用，以在
    模型包含 `Mul` 节点时触发失败：

    ```bash
    polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect \
        --fail-regex "Op: Mul" \
        --check polygraphy inspect model polygraphy_debug.onnx --show layers
    ```

4.  **[可选]** 作为健全性检查，我们可以检查我们简化的模型，以确保它确实包含 `Mul` 节点：

    ```bash
    polygraphy inspect model initial_reduced.onnx --show layers
    ```

5.  由于我们在上一步中使用了 `bisect` 模式，模型可能没有尽可能地小。
    为了进一步完善它，我们将再次在 `linear` 模式下运行 `debug reduce`：

    ```bash
    polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear --load-inputs layerwise_inputs.json \
        --check polygraphy run polygraphy_debug.onnx --trt \
                --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json
    ```

    **模拟失败：** 我们将使用与之前相同的技术：

    ```bash
    polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear \
        --fail-regex "Op: Mul" \
        --check polygraphy inspect model polygraphy_debug.onnx --show layers
    ```

6.  **[可选]** 在这个阶段，`final_reduced.onnx` 应该只包含失败的节点 - `Mul`。
    我们可以用 `inspect model` 来验证这一点：

    ```bash
    polygraphy inspect model final_reduced.onnx --show layers
    ```


## 进一步阅读

-   有关 `debug` 工具如何工作的更多详细信息，请参阅帮助输出：
    `polygraphy debug -h` 和 `polygraphy debug reduce -h`。

-   另请参阅 [`debug reduce` 操作指南](../../../../how-to/use_debug_reduce_effectively.md)
    以获取更多信息、提示和技巧。
