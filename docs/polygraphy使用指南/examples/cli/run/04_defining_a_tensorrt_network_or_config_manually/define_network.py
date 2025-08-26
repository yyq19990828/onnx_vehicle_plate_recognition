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
解析一个 ONNX 模型，然后用一个恒等层扩展它。
"""
from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath

parse_onnx = NetworkFromOnnxPath("identity.onnx")


# 如果我们定义一个名为 `load_network` 的函数，polygraphy 可以
# 直接使用它来代替使用模型文件。
#
# 提示：如果我们的函数不叫 `load_network`，我们可以用模型参数显式指定
# 名称，用冒号分隔。例如，`define_network.py:my_func`。
@func.extend(parse_onnx)
def load_network(builder, network, parser):
    # 注意：func.extend() 使此函数的签名为 `() -> (builder, network, parser)`
    # 有关其工作原理的详细信息，请参阅 examples/api/03_interoperating_with_tensorrt

    # 向网络附加一个恒等层
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)

    output = network.add_identity(prev_output).get_output(0)
    network.mark_output(output)

    # 注意，我们不需要返回任何东西 - `extend()` 会为我们处理！
