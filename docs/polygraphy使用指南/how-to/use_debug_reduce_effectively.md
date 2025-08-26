# 有效使用 `debug reduce`


## 目录

- [简介](#introduction)
- [关于具有动态输入形状的模型的说明](#a-note-on-models-with-dynamic-input-shapes)
- [调试精度错误](#debugging-accuracy-errors)
    - [容差](#tolerances)
    - [生成黄金值](#generating-golden-values)
- [提示和技巧](#tips-and-tricks)
    - [保存中间模型](#saving-intermediate-models)
    - [从最小良好模型中获得的见解](#insights-from-minimum-good-models)
    - [缩减模式](#reduction-modes)
- [更多阅读](#further-reading)


## 简介

`debug reduce` 子工具允许您迭代地缩减失败的 ONNX 模型，以找到最小的失败案例，这可能比原始模型更容易调试。`debug reduce` 的基本步骤如下：

1. 从原始图中删除一些节点，并将新模型写入 `polygraphy_debug.onnx`（此路径可以使用 `--iter-artifact` 选项更改）。

2. 以交互方式或在提供 `--check` 命令时自动评估模型。

3. 如果模型仍然失败，则删除更多节点，否则将节点添加回来；然后，重复该过程。

本指南提供了一些通用信息以及有关如何有效使用 `debug reduce` 的提示和技巧。

另请参阅 [`debug` 子工具的通用操作指南](./use_debug_subtools_effectively.md)，其中包含适用于所有 `debug` 子工具的信息。


## 关于具有动态输入形状的模型的说明

对于具有动态输入形状的模型，您可能并不总是知道模型中所有中间张量的形状。因此，当您检查子图时，您最终可能会使用不正确的张量形状。

有两种方法可以解决这个问题：

1. 使用 `polygraphy surgeon sanitize --override-input-shapes <shapes>` 来冻结模型中的输入形状
2. 向 `debug reduce` 提供 `--model-input-shapes`，它将使用形状推断来推断中间张量的形状。

如果您的模型使用形状操作，通常最好使用选项 (1) 并使用 `--fold-constants` 折叠形状操作。

在任何一种情况下，如果形状推断存在问题，您都可以使用 `--force-fallback-shape-inference` 来通过运行推理来推断形状。

或者，您可以使用 `--no-reduce-inputs`，这样就不会修改模型输入。在每次迭代期间生成的 `polygraphy_debug.onnx` 子图将始终使用原始模型的输入；只会从末尾删除层。


## 调试精度错误

精度错误调试起来尤其复杂，因为图中早期层引入的错误可能会被后续层放大，从而难以确定哪个层是错误的真正根源。本节概述了在使用 `debug reduce` 调试精度错误时需要牢记的一些事项。

### 容差

在某些模型架构中，中间层可能会有很大的误差，而不一定会导致最终模型输出的精度问题。因此，请确保您用于比较的容差足够高，以忽略这些类型的误报。

同时，容差必须足够低才能捕获真正的错误。

一个好的起点是将容差设置得接近您在完整模型中观察到的误差。


### 生成黄金值

在生成用于比较的黄金值时，您可以采用两种不同的方法，每种方法都有其自身的优缺点：

1. **提前为所有层生成黄金值。**

    在提前生成黄金值时，您需要确保每个子图的输入值都来自黄金值。否则，将子图的输出与黄金值进行比较将毫无意义。有关此方法的详细信息，请参阅[示例](../examples/cli/debug/02_reducing_failing_onnx_models/)。

2. **为每个子图生成黄金值。**

    为每个子图重新生成黄金值可能需要较少的手动工作，但缺点是它不一定能准确地复制子图在更大图的上下文中的行为。例如，如果模型中的错误是由原始模型中间层的溢出引起的，那么为每个子图生成新的输入值可能无法复现它。


## 提示和技巧


### 保存中间模型

在某些情况下，能够访问在缩减过程中生成的每个模型是很有用的。这样，如果缩减提前退出或未能生成最小模型，您仍然有一些东西可以使用。此外，您可以手动比较各种通过和失败的子图以识别模式，这可能有助于您确定错误的根本原因。

您可以向 `debug reduce` 指定 `--artifacts polygraphy_debug.onnx`，以自动将每次迭代中的模型分类到 `good` 和 `bad` 目录中。文件名将包含迭代次数，以便您可以轻松地将其与缩减期间的日志输出相关联。


### 从最小良好模型中获得的见解

除了最小失败模型之外，`debug reduce` 还可以生成最小通过模型。通常，这是与最小失败模型大小最接近的通过模型。将其与最小失败模型进行比较可以为了解故障的根本原因提供额外的见解。

要使 `debug reduce` 保存最小通过模型，请使用 `--min-good <path>` 选项。


### 缩减模式

`debug reduce` 提供了多种缩减模型的策略，您可以使用 `--mode` 选项指定：`bisect` 以 `O(log(N))` 时间运行，而 `linear` 以 `O(N)` 时间运行，但可能会导致更小的模型。一个好的折衷方案是在原始模型上使用 `bisect`，然后使用 `linear` 进一步缩减结果。


## 更多阅读

- [`debug` 子工具的操作指南](./use_debug_subtools_effectively.md)，其中包含适用于所有 `debug` 子工具的信息。

- [`debug reduce` 示例](../examples/cli/debug/02_reducing_failing_onnx_models/)，它演示了此处概述的一些功能。