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
这个脚本演示了如何将 Polygraphy 与后端提供的 API 结合使用。
具体来说，在这种情况下，我们使用 TensorRT API 来打印网络名称并启用 FP16 模式。
"""
import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)


# 提示：立即评估的功能性 API 使得与 TensorRT 等后端的互操作变得非常容易。
# 有关详细信息，请参阅示例 06 (`examples/api/06_immediate_eval_api`)。

# 我们可以使用 `extend` 装饰器轻松扩展 Polygraphy 提供的延迟加载器
# 我们装饰的函数所带的参数应与我们正在扩展的加载器的返回值相匹配。


# 对于 `NetworkFromOnnxPath`，我们可以从 API 文档中看到它返回一个 TensorRT
# 构建器、网络和解析器。这就是我们的函数将接收的内容。
@func.extend(NetworkFromOnnxPath("identity.onnx"))
def load_network(builder, network, parser):
    # 在这里我们可以修改网络。对于这个例子，我们只设置网络名称。
    network.name = "MyIdentity"
    print(f"网络名称: {network.name}")

    # 注意，我们不需要返回任何东西 - `extend()` 会为我们处理！


# 如果 Polygraphy 中缺少某个构建器配置选项，我们可以使用 TensorRT API 轻松设置它。
# 我们的函数将接收一个 TensorRT IBuilderConfig，因为这是 `CreateConfig` 返回的内容。
@func.extend(CreateConfig())
def load_config(config):
    # Polygraphy 支持 fp16 标志，但如果不支持，我们可以这样做：
    config.set_flag(trt.BuilderFlag.FP16)


def main():
    # 由于我们不再需要 TensorRT API，我们可以回到常规的 Polygraphy。
    #
    # 注意：由于我们使用的是延迟加载器，我们将函数作为参数提供 - 我们自己*不*调用它们。
    build_engine = EngineFromNetwork(load_network, config=load_config)

    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # 注意: 运行器拥有输出缓冲区，并可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果要存储多次推理的结果，应使用 `copy.deepcopy()`。
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # 这是一个 identity 模型！

        print("推理成功！")


if __name__ == "__main__":
    main()
