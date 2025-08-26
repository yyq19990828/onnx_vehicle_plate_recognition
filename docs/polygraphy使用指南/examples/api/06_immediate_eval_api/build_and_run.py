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
这个脚本使用 Polygraphy 的立即评估功能性 API
来加载一个 ONNX 模型，将其转换为 TensorRT 网络，在网络末尾添加一个 identity
层，构建一个启用 FP16 模式的引擎，
保存引擎，最后运行推理。
"""
import numpy as np
from polygraphy.backend.trt import (
    TrtRunner,
    create_config,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)


def main():
    # 在 Polygraphy 中，如果对象是通过可调用对象的返回值提供的，
    # 则加载器和运行器将获得对象的所有权。例如，当我们使用延迟加载器时，
    # 我们不需要担心对象的生命周期。
    #
    # 由于我们是立即评估，我们获得了对象的所有权，并负责释放它们。
    builder, network, parser = network_from_onnx_path("identity.onnx")

    # 使用 identity 层扩展网络（纯粹为了示例）。
    #   请注意，与延迟加载器不同，我们不需要做任何特殊的事情来修改网络。
    #   如果我们使用延迟加载器，我们将需要使用 `func.extend()`，如示例 03 和示例 05 中所述。
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)
    output = network.add_identity(prev_output).get_output(0)
    output.name = "output"
    network.mark_output(output)

    # 创建一个 TensorRT IBuilderConfig，以便我们可以构建启用 FP16 的引擎。
    config = create_config(builder, network, fp16=True)

    engine = engine_from_network((builder, network), config)

    # 要在其他地方重用引擎，我们可以将其序列化并保存到文件中。
    save_engine(engine, path="identity.engine")

    with TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # 注意: 运行器拥有输出缓冲区，并可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果要存储多次推理的结果，应使用 `copy.deepcopy()`。
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["output"], inp_data)  # 这是一个 identity 模型！

        print("推理成功！")


if __name__ == "__main__":
    main()
