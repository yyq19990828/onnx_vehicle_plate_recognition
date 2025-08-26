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
该脚本演示如何使用 Polygraphy 提供的 Calibrator API 
来校准 TensorRT 引擎以在 INT8 精度下运行。
"""
import numpy as np
from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)
from polygraphy.logger import G_LOGGER


# `Calibrator` 的数据加载器参数可以是任何可迭代对象或生成器，用于生成 `feed_dict`。
# `feed_dict` 只是输入名称到相应输入的映射。
def calib_data():
    for _ in range(4):
        # 提示：如果您的校准数据已经在 GPU 上，您可以提供 GPU 指针
        # （作为 `int`）、Polygraphy `DeviceView` 或 PyTorch 张量，而不是 NumPy 数组。
        #
        # 有关 `DeviceView` 的详细信息，请参阅 `polygraphy/cuda/cuda.py`。
        yield {"x": np.ones(shape=(1, 1, 2, 2), dtype=np.float32)}  # 完全真实的数据


def main():
    # 如果我们想要缓存校准数据，我们可以提供路径或类文件对象。
    # 这让我们在下次构建引擎时避免运行校准。
    #
    # 提示：您可以直接将此校准器与 TensorRT API 一起使用（例如 config.int8_calibrator）。
    # 如果您不想，您不必将其与 Polygraphy 加载器一起使用。
    calibrator = Calibrator(data_loader=calib_data(), cache="identity-calib.cache")

    # 除了提供校准器外，我们还必须启用 int8 模式。
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"),
        config=CreateConfig(int8=True, calibrator=calibrator),
    )

    # 当我们激活运行器时，它将校准并构建引擎。如果我们想要
    # 看到来自 TensorRT 的日志输出，我们可以临时提高日志详细程度：
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        # 最后，我们可以使用一些虚拟输入数据测试我们的 int8 TensorRT 引擎：
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # 注意：运行器拥有输出缓冲区，可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果您想存储多个推理的结果，应该使用 `copy.deepcopy()`。
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # 这是一个恒等模型！


if __name__ == "__main__":
    main()
