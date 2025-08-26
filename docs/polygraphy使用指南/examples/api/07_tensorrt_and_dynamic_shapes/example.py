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
这个脚本构建了一个包含 3 个独立优化配置文件的引擎，每个配置文件
都针对特定的用例构建。然后，它创建 3 个独立的执行上下文
和相应的 `TrtRunner` 用于推理。
"""
import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER


def main():
    # Profile 将每个输入张量映射到一个形状范围。
    # `add()` 方法可用于为单个输入添加形状。
    #
    # 提示：为了节省行数，可以链接对 `add` 的调用：
    #     profile.add("input0", ...).add("input1", ...)
    #
    #   当然，您也可以这样写：
    #     profile.add("input0", ...)
    #     profile.add("input1", ...)
    #
    profiles = [
        # 低延迟情况。为获得最佳性能，min == opt == max。
        Profile().add("X", min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)),
        # 动态批处理情况。我们使用 `4` 作为 opt 批处理大小，因为这是我们最常见的情况。
        Profile().add("X", min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)),
        # 离线情况。为获得最佳性能，min == opt == max。
        Profile().add(
            "X", min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)
        ),
    ]

    # 有关立即评估的功能加载器（如 `engine_from_network`）的详细信息，请参阅 examples/api/06_immediate_eval_api。
    # 请注意，我们可以自由混合延迟加载器和立即评估加载器。
    engine = engine_from_network(
        network_from_onnx_path("dynamic_identity.onnx"),
        config=CreateConfig(profiles=profiles),
    )

    # 我们将保存引擎，以便可以使用 `inspect model` 对其进行检查。
    # 这应该可以很容易地看到引擎绑定的布局方式。
    save_engine(engine, "dynamic_identity.engine")

    # 我们将创建但*不*激活三个独立的运行器，每个运行器都有一个独立的上下文。
    #
    # 提示：通过直接提供上下文，而不是通过延迟加载器，
    # 我们可以确保运行器*不会*获得它的所有权。
    #
    low_latency = TrtRunner(engine.create_execution_context())

    # 注意：以下两行可能会导致 TensorRT 显示错误，因为配置文件 0
    # 已被第一个执行上下文使用。我们将使用 G_LOGGER.verbosity() 抑制它们。
    #
    with G_LOGGER.verbosity(G_LOGGER.CRITICAL):
        # 我们可以使用运行器的 `optimization_profile` 参数来确保使用正确的优化配置文件。
        # 这消除了以后调用 `set_profile()` 的需要。
        dynamic_batching = TrtRunner(
            engine.create_execution_context(), optimization_profile=1
        )  # 使用第二个配置文件，该配置文件用于动态批处理。

        # 为了举例，我们*不会*在这里使用 `optimization_profile`。
        # 相反，我们将在激活运行器后使用 `set_profile()`。
        offline = TrtRunner(engine.create_execution_context())

    # 最后，我们可以根据需要激活运行器。
    #
    # 注意：由于上下文和引擎已经创建，运行器在激活期间只需要
    # 分配输入和输出缓冲区。

    input_img = np.ones((1, 3, 28, 28), dtype=np.float32)  # 一个输入“图像”

    with low_latency:
        outputs = low_latency.infer({"X": input_img})
        assert np.array_equal(outputs["Y"], input_img)  # 这是一个 identity 模型！

        print("低延迟运行器成功！")

        # 在我们在线处理请求时，我们可能会决定我们需要动态批处理
        # 一会儿。
        #
        # 注意：我们假设在这里激活运行器会很便宜，所以我们可以及时
        # 启动动态批处理运行器。
        #
        # 提示：如果激活运行器不便宜（例如输入/输出缓冲区很大），
        # 最好一直保持运行器处于活动状态。
        #
        with dynamic_batching:
            # 我们将通过重复我们的假输入图像来创建假批次。
            small_input_batch = np.repeat(input_img, 4, axis=0)  # 形状：(4, 3, 28, 28)
            outputs = dynamic_batching.infer({"X": small_input_batch})
            assert np.array_equal(outputs["Y"], small_input_batch)

    # 如果我们稍后再次需要动态批处理，我们可以再次激活运行器。
    #
    # 注意：这一次，我们*不*需要设置配置文件。
    #
    with dynamic_batching:
        # 注意：我们可以使用配置文件范围内的任何形状，而无需
        # 额外的设置 - Polygraphy 在幕后处理细节！
        #
        large_input_batch = np.repeat(input_img, 16, axis=0)  # 形状：(16, 3, 28, 28)
        outputs = dynamic_batching.infer({"X": large_input_batch})
        assert np.array_equal(outputs["Y"], large_input_batch)

        print("动态批处理运行器成功！")

    with offline:
        # 注意：当我们第一次激活此运行器时，我们需要设置配置文件索引（默认为 0）。
        # 由于我们在创建运行器时提供了自己的执行上下文，我们只需要*一次*执行此操作。
        # 我们的设置会保留，因为即使在运行器停用后，上下文仍将保持活动状态。
        # 如果我们让运行器拥有上下文，我们每次激活运行器时都需要重复此步骤。
        #
        # 或者，我们可以使用 `optimization_profile` 参数（见上文）。
        #
        offline.set_profile(
            2
        )  # 使用第三个配置文件，该配置文件用于离线情况。

        large_offline_batch = np.repeat(
            input_img, 128, axis=0
        )  # 形状：(128, 3, 28, 28)
        outputs = offline.infer({"X": large_offline_batch})
        assert np.array_equal(outputs["Y"], large_offline_batch)

        print("离线运行器成功！")


if __name__ == "__main__":
    main()
