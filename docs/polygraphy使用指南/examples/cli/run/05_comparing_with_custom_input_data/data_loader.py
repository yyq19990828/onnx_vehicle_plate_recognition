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
演示了在 Polygraphy 中加载自定义输入数据的两种方法：

选项 1：定义一个 `load_data` 函数，该函数返回一个生成器，该生成器产生
    feed_dicts，以便此脚本可用作
    --data-loader-script 命令行参数的参数。

选项 2：将输入数据写入一个 JSON 文件，该文件可用作
    --load-inputs 命令行参数的参数。
"""
import numpy as np
from polygraphy.json import save_json

INPUT_SHAPE = (1, 2, 28, 28)


# 选项 1：定义一个将产生 feed_dicts (即 Dict[str, np.ndarray]) 的函数
def load_data():
    for _ in range(5):
        yield {
            "x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)
        }  # 仍然是完全真实的数据


# 选项 2：使用 `save_json()` 帮助程序创建一个包含输入数据的 JSON 文件。
#   `save_json()` 的输入应为类型：List[Dict[str, np.ndarray]]。
#   为方便起见，我们将重用我们的 `load_data()` 实现来生成列表。
input_data = list(load_data())
save_json(input_data, "custom_inputs.json", description="自定义输入数据")
