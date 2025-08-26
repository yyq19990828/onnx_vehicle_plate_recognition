# 使用自定义输入数据

对于任何使用推理输入数据的工具，例如 `run` 或 `convert`，Polygraphy
提供了 2 种提供自定义输入数据的方法：

1. `--load-inputs`/`--load-input-data`，它接受一个包含
    `List[Dict[str, np.ndarray]]` 的 JSON 文件的路径。
    该 JSON 文件应使用 Polygraphy 的 JSON 实用程序（例如 `save_json`）在
    `polygraphy.json` 子模块中创建。

    *注意：这将导致 Polygraphy 将整个对象加载到内存中，因此如果数据非常大，*
        *这可能不切实际或不可能。*

2. `--data-loader-script`，它接受一个 Python 脚本的路径，该脚本定义了一个 `load_data` 函数，
    该函数返回一个数据加载器。数据加载器可以是任何产生
    `Dict[str, np.ndarray]` 的可迭代对象或生成器。通过使用生成器，我们可以避免一次性加载所有数据，
    而是将其限制为一次只加载一个输入。

    *提示：如果您有一个现有的脚本已经定义了这样的函数，您**不**需要*
        *仅仅为了 `--data-loader-script` 而创建一个单独的脚本。您可以简单地使用现有的脚本*
        *并指定函数的名称（如果它不是 `load_data`）*


## 更多阅读

- 有关上述两种方法的示例，请参阅 [`run` 示例 05](../examples/cli/run/05_comparing_with_custom_input_data/)。
