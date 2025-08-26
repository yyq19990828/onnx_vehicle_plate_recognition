# 在 onnx 模型中匹配并用插件替换子图

## 简介

`plugin` 工具提供子工具来查找和替换 onnx 模型中的子图。

子图替换是一个三步过程：

1.  根据插件的图模式 (pattern.py) 查找匹配的子图，并在用户可编辑的中间文件 (config.yaml) 中列出潜在的替换
2.  审查并（如有必要）编辑潜在替换列表 (config.yaml)
3.  根据潜在替换列表 (config.yaml) 用插件替换子图

`original.onnx` -------> `match` -------> `config.yaml` -------> `replace` -------> `replaced.onnx`
`plugins` ----------------^ `usr input`---^ `plugins`--------^

## 详细信息

### 匹配

在模型中查找匹配的子图是基于插件提供的图模式描述 (`pattern.py`) 完成的。
图模式描述 (`pattern.py`) 包含有关图节点的拓扑和附加约束的信息，以及一种基于匹配子图计算插件属性的方法。
只有提供图模式描述 (pattern.py) 的插件才会被考虑用于匹配。

匹配的结果存储在一个名为 `config.yaml` 的中间文件中。
用户应审查和编辑此文件，因为它作为替换步骤的待办事项列表。例如，如果有 2 个匹配的子图，但只有一个应该被替换，则可以从文件中删除该结果。

作为预览/试运行步骤，`plugin list` 子工具可以显示潜在替换的列表，而无需生成中间文件。

### 替换

用插件替换子图使用在匹配阶段生成的 `config.yaml` 文件。此文件中列出的任何匹配子图都将被删除，并替换为表示插件的单个节点。原始文件被保留，并保存一个新文件，其中进行了替换。默认情况下，此文件名为 `replaced.onnx`。

### 比较

可以比较原始模型和替换后的模型，以检查它们在插件替换前后的行为是否相同：
`polygraphy run original.onnx --trt --save-outputs model_output.json`
`polygraphy run replaced.onnx --trt --load-outputs model_output.json`

## 运行示例

1.  在示例网络中查找并保存 toyPlugin 的匹配项：

    ```bash
    polygraphy plugin match toy_subgraph.onnx \
        --plugin-dir ./plugins -o config.yaml
    ```

    <!-- Polygraphy Test: Ignore Start -->

    这将显示如下内容：

    ```
    checking toyPlugin in model
    [I] Start a subgraph matching...
    [I] 	Checking node: n1 against pattern node: Anode.
    [I] 	No match because: Op did not match. Node op was: O but pattern op was: A.
    [I] Start a subgraph matching...
    [I] Found a matched subgraph!
    [I] Start a subgraph matching...
    ```

    生成的 config.yaml 将如下所示：

    ```
    name: toyPlugin
    instances:
    - inputs:
    - i1
    - i1
    outputs:
    - o1
    - o2
    attributes:
        x: 1
    ```

    <!-- Polygraphy Test: Ignore End -->

2.  **[可选]** 在示例网络中列出 toyPlugin 的匹配项，而不保存 config.yaml：

    ```bash
    polygraphy plugin list toy_subgraph.onnx \
        --plugin-dir ./plugins
    ```

    <!-- Polygraphy Test: Ignore Start -->

    这将显示如下内容：

    ```
    checking toyPlugin in model
    [I] Start a subgraph matching...
    [I] 	Checking node: n1 against pattern node: Anode.
    [I] 	No match because: Op did not match. Node op was: O but pattern op was: A.
    [I] Start a subgraph matching...
    ...
    [I] Found a matched subgraph!
    [I] Start a subgraph matching...
    [I] 	Checking node: n6 against pattern node: Anode.
    [I] 	No match because: Op did not match. Node op was: E but pattern op was: A.
    the following plugins would be used:
    {'toyPlugin': 1}
    ```

    不会生成 config.yaml，因为此命令仅用于打印每个插件的匹配数
    <!-- Polygraphy Test: Ignore End -->

`plugin replace` 子工具用插件替换 onnx 模型中的子图

3.  用 toyPlugin 替换示例网络的一部分：

    ```bash
    polygraphy plugin replace toy_subgraph.onnx \
        --plugin-dir ./plugins --config config.yaml -o replaced.onnx
    ```

    <!-- Polygraphy Test: Ignore Start -->

    这将显示如下内容：

    ```
    [I] Loading model: toy_subgraph.onnx
    ```

    结果文件是 replaced.onnx，其中示例网络中的一个子图被 toyPlugin 替换
    <!-- Polygraphy Test: Ignore End -->
