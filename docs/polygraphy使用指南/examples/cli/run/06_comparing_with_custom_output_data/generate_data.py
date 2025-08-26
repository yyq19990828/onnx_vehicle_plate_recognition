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
为恒等模型生成输入和输出数据并将其保存到磁盘。
"""
import numpy as np
from polygraphy.comparator import RunResults
from polygraphy.json import save_json

INPUT_SHAPE = (1, 1, 2, 2)


# 我们将生成任意输入数据，然后在将两者保存到磁盘之前“计算”预期的输出数据。
# 为了让 Polygraphy 加载输入和输出数据，它们必须采用以下格式：
#   - 输入数据：List[Dict[str, np.ndarray]]（feed_dicts 列表）
#   - 输出数据：RunResults


# 生成与模型兼容的任意输入数据。
#
# 提示：我们也可以像在 `run` 示例 05 (05_comparing_with_custom_input_data) 中那样使用生成器。
#   在这种情况下，我们只需将此脚本提供给 `--data-loader-script`，而不是在此处保存输入
#   然后使用 `--load-inputs`。
input_data = {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}

# 注意：输入数据必须在列表中（以支持多组输入），因此我们在保存之前创建一个。
#   `description` 参数是可选的：
save_json([input_data], "custom_inputs.json", description="自定义输入数据")


# 根据输入数据“计算”输出。由于这是一个恒等模型，我们可以直接复制输入。
output_data = {"y": input_data["x"]}

# 要保存输出数据，我们可以创建一个 RunResults 对象：
custom_outputs = RunResults()

# `add()` 辅助函数使我们可以轻松添加条目。
#
# 注意：与输入数据一样，输出数据也必须在列表中，因此我们在保存之前创建一个。
#
# 提示：或者，我们可以使用如下方法手动添加条目：
#   runner_name = "custom_runner"
#   custom_outputs[runner_name] = [IterationResult(output_data, runner_name=runner_name), ...]
#
# 提示：要存储来自多个不同实现的输出，您可以为 `add()` 指定不同的 `runner_name`。
#   如果省略 `runner_name`，则使用默认值。
custom_outputs.add([output_data], runner_name="custom_runner")
custom_outputs.save("custom_outputs.json")
