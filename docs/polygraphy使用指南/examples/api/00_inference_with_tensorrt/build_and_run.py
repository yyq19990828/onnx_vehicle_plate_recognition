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
这个脚本从一个 ONNX identity 模型开始，
构建并运行一个启用了 FP16 精度的 TensorRT 引擎。
"""
import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
    TrtRunner,
)


def main():
    # 我们可以将多个延迟加载器组合在一起以获得所需的转换。
    # 在本例中，我们想要 ONNX -> TensorRT 网络 -> TensorRT 引擎 (使用 fp16)。
    #
    # 注意: `build_engine` 是一个返回引擎的 *可调用对象*，而不是引擎本身。
    #   要直接获取引擎，您可以使用立即评估的功能性 API。
    #   有关详细信息，请参阅 examples/api/06_immediate_eval_api。
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"), config=CreateConfig(fp16=True)
    )  # 注意，config 是一个可选参数。

    # 要在其他地方重用引擎，我们可以将其序列化并保存到文件中。
    # `SaveEngine` 延迟加载器在被调用时将返回 TensorRT 引擎，
    # 这允许我们将其与其他加载器链接在一起。
    build_engine = SaveEngine(build_engine, path="identity.engine")

    # 一旦我们的加载器准备就绪，推理就只是构建一个运行器，
    # 使用上下文管理器 (即 `with TrtRunner(...)`) 激活它并调用 `infer()` 的问题。
    #
    # 注意: 您可以使用 activate() 函数代替上下文管理器，但您需要确保
    # 调用 deactivate() 以避免内存泄漏。因此，上下文管理器是更安全的选择。
    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # 注意: 运行器拥有输出缓冲区，并可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果要存储多次推理的结果，应使用 `copy.deepcopy()`。
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # 这是一个 identity 模型！

        print("推理成功！")


if __name__ == "__main__":
    main()
