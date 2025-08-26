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
这个脚本演示了如何使用示例 03 中介绍的 extend() API
来使用 TensorRT 网络 API 构建 TensorRT 网络。
"""
import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateNetwork, EngineFromNetwork, TrtRunner


INPUT_NAME = "input"
INPUT_SHAPE = (64, 64)
OUTPUT_NAME = "output"


# 就像在示例 03 中一样，我们可以使用 `extend` 将我们自己的功能添加到现有的延迟加载器中。
# `CreateNetwork` 将创建一个空网络，然后我们可以自己填充它。
@func.extend(CreateNetwork())
def create_network(builder, network):
    # 这个网络会将输入张量加 1。
    inp = network.add_input(name=INPUT_NAME, shape=INPUT_SHAPE, dtype=trt.float32)
    ones = network.add_constant(
        shape=INPUT_SHAPE, weights=np.ones(shape=INPUT_SHAPE, dtype=np.float32)
    ).get_output(0)
    add = network.add_elementwise(
        inp, ones, op=trt.ElementWiseOperation.SUM
    ).get_output(0)
    add.name = OUTPUT_NAME
    network.mark_output(add)

    # 注意，我们不需要返回任何东西 - `extend()` 会为我们处理！


def main():
    # 构建网络后，我们可以回到使用常规的 Polygraphy API。
    #
    # 注意：由于我们使用的是延迟加载器，我们将 `create_network` 函数作为
    # 参数提供 - 我们自己*不*调用它。
    build_engine = EngineFromNetwork(create_network)

    with TrtRunner(build_engine) as runner:
        feed_dict = {
            INPUT_NAME: np.random.random_sample(INPUT_SHAPE).astype(np.float32)
        }

        # 注意: 运行器拥有输出缓冲区，并可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果要存储多次推理的结果，应使用 `copy.deepcopy()`。
        outputs = runner.infer(feed_dict)

        assert np.array_equal(outputs[OUTPUT_NAME], (feed_dict[INPUT_NAME] + 1))

        print("推理成功！")


if __name__ == "__main__":
    main()
