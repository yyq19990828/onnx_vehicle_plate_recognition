#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
这个脚本演示了如何使用 `load_json` 和 `RunResults` API 分别加载
和操作推理输入和输出。
"""

from polygraphy.comparator import RunResults
from polygraphy.json import load_json


def main():
    # 使用 `load_json` API 从文件加载输入。
    #
    # 注意：`save_json` 和 `load_json` 独立帮助程序应仅与非 Polygraphy 对象一起使用。
    # 支持序列化的 Polygraphy 对象包括 `save` 和 `load` 方法。
    inputs = load_json("inputs.json")

    # 输入存储为 `List[Dict[str, np.ndarray]]`，即 feed_dicts 列表，
    # 其中每个 feed_dict 将输入名称映射到 NumPy 数组。
    #
    # 提示：在典型情况下，我们只有一个迭代，所以我们只看第一项。
    # 如果您需要访问来自多个迭代的输入，您可以这样做：
    #
    #    for feed_dict in inputs:
    #        for name, array in feed_dict.items():
    #            ... # 在这里对输入做一些事情
    #
    [feed_dict] = inputs
    for name, array in feed_dict.items():
        print(f"输入: '{name}' | 值:\n{array}")

    # 使用 `RunResults.load` API 从文件加载结果。
    #
    # 提示：您可以在此处提供文件路径或类文件对象。
    results = RunResults.load("outputs.json")

    # `RunResults` 对象的结构类似于 `Dict[str, List[IterationResult]]``，
    # 将运行器名称映射到一个或多个迭代的推理输出。
    # `IterationResult` 的行为就像一个 `Dict[str, np.ndarray]`，将输出名称映射到
    # NumPy 数组。
    #
    # 提示：在典型情况下，我们只有一个迭代，所以我们可以直接在循环中解包它。
    # 如果您需要访问来自多个迭代的输出，您可以这样做：
    #
    #    for runner_name, iters in results.items():
    #        for outputs in iters:
    #             ... # 在这里对输出做一些事情
    #
    for runner_name, [outputs] in results.items():
        print(f"\n处理运行器的输出: {runner_name}")
        # 现在您可以读取或修改每个运行器的输出。
        # 为了这个例子，我们只打印它们：
        for name, array in outputs.items():
            print(f"输出: '{name}' | 值:\n{array}")


if __name__ == "__main__":
    main()
