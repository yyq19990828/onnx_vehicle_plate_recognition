# 调试 TensorRT 精度问题

TensorRT 中的精度问题，尤其是在大型网络中，调试起来可能具有挑战性。
使它们易于管理的一种方法是减小问题规模或查明故障源。

本指南旨在提供一种通用的方法；它被组织成一个扁平的流程图——
在每个分支处，都提供了两个链接，以便您可以选择最适合您情况的链接。

如果您使用的是 ONNX 模型，请在继续之前尝试[清理它](../examples/cli/surgeon/02_folding_constants/)，因为这在某些情况下可能会解决问题。


## 真实输入数据有影响吗？

某些模型可能对输入数据敏感。例如，真实输入可能会比随机生成的输入产生更好的精度。Polygraphy 提供了多种提供真实输入数据的方法，在 [`run` 示例 05](../examples/cli/run/05_comparing_with_custom_input_data/) 中有概述。

使用真实输入数据是否可以提高精度？

- 是的，使用真实输入数据时精度可以接受。

    这很可能意味着没有错误；相反，您的模型对输入数据敏感。

- 不，即使使用真实输入数据，我仍然看到精度问题。

    转到：[间歇性还是非间歇性？](#intermittent-or-not)


## 间歇性还是非间歇性？

引擎构建之间的这个问题是间歇性的吗？

- 是的，有时当我重建引擎时，精度问题会消失。

    转到：[调试间歇性精度问题](#debugging-intermittent-accuracy-issues)

- 不，我每次构建引擎时都会看到精度问题。

    转到：[逐层分析是一个选项吗？](#is-layerwise-an-option)


## 调试间歇性精度问题

由于引擎构建过程是不确定的，因此每次构建引擎时可能会选择不同的策略（即层实现）。当其中一种策略出现故障时，这可能会表现为间歇性故障。Polygraphy 包含一个 `debug build` 子工具来帮助您找到此类策略。

有关更多信息，请参阅 [`debug` 示例 01](../examples/cli/debug/01_debugging_flaky_trt_tactics/)。

您能找到失败的策略吗？

- 是的，我知道哪个策略有问题。

    转到：[您有一个最小的失败案例！](#you-have-a-minimal-failing-case)

- 不，故障可能不是间歇性的。

    转到：[逐层分析是一个选项吗？](#is-layerwise-an-option)


## 逐层分析是一个选项吗？

如果精度问题可以持续复现，那么最好的下一步是找出是哪个层导致了故障。Polygraphy 包含一种将网络中的所有张量标记为输出的机制，以便可以对它们进行比较；但是，这可能会影响 TensorRT 的优化过程。因此，我们需要确定当所有输出张量都被标记时，我们是否仍然观察到精度问题。

在继续之前，请参阅[此示例](../examples/cli/run/01_comparing_frameworks/README.md#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt)以了解如何比较逐层输出的详细信息。

在比较逐层输出时，您是否能够复现精度故障？

- 是的，即使我标记了网络中的其他输出，故障也会复现。

    转到：[提取失败的子图](#extracting-a-failing-subgraph)

- 不，标记其他输出会导致精度提高，或者当我标记其他输出时，我根本无法运行模型。

    转到：[缩减失败的 Onnx 模型](#reducing-a-failing-onnx-model)


## 提取失败的子图

由于我们能够比较逐层输出，我们应该能够通过查看输出比较日志来确定哪个层首先引入了错误。一旦我们知道哪个层有问题，我们就可以从模型中提取它。

为了找出相关层的输入和输出张量，我们可以使用 `polygraphy inspect model`。有关详细信息，请参阅以下示例之一：

- [TensorRT 网络](../examples/cli/inspect/01_inspecting_a_tensorrt_network/)
- [ONNX 模型](../examples/cli/inspect/03_inspecting_an_onnx_model/)。

接下来，我们可以提取一个仅包含问题层的子图。
有关更多信息，请参阅 [`surgeon` 示例 01](../examples/cli/surgeon/01_isolating_subgraphs/)。

这个孤立的子图是否复现了问题？

- 是的，子图也失败了。

    转到：[您有一个最小的失败案例！](#you-have-a-minimal-failing-case)

- 不，子图工作正常。

    转到：[缩减失败的 Onnx 模型](#reducing-a-failing-onnx-model)


## 缩减失败的 ONNX 模型

当我们无法使用逐层比较来查明故障源时，我们可以使用一种暴力方法来缩减 ONNX 模型——迭代地生成越来越小的子图，以找到仍然失败的最小可能子图。`debug reduce` 工具有助于自动化此过程。

有关更多信息，请参阅 [`debug` 示例 02](../examples/cli/debug/02_reducing_failing_onnx_models/)。

缩减后的模型是否失败？

- 是的，缩减后的模型失败了。

    转到：[您有一个最小的失败案例！](#you-have-a-minimal-failing-case)

- 不，缩减后的模型没有失败，或者以不同的方式失败。

    转到：[仔细检查您的缩减选项](#double-check-your-reduce-options)


## 仔细检查您的缩减选项

如果缩减后的模型不再失败，或者以不同的方式失败，请确保您的 `--check` 命令是正确的。您可能还想使用 `--fail-regex` 来确保在缩减模型时只考虑精度故障（而不是其他无关的故障）。

- 再次尝试缩减。

    转到：[缩减失败的 Onnx 模型](#reducing-a-failing-onnx-model)

## 您有一个最小的失败案例！

如果您已经到了这一步，那么您现在就有一个最小的失败案例了！进一步的调试应该会容易得多。

如果您是 TensorRT 开发人员，此时您需要深入研究代码。
如果不是，请报告您的错误！
