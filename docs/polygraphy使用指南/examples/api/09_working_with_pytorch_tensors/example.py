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
该脚本演示如何将 PyTorch 张量与 TensorRT 运行器和校准器一起使用。
"""

import torch

from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
)

# 如果您的 PyTorch 安装支持 GPU，那么我们将直接在 GPU 内存中分配张量。
# 这意味着校准器和运行器可以跳过我们在使用 NumPy 数组时会产生的
# 主机到设备的复制。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calib_data():
    for _ in range(4):
        yield {"x": torch.ones((1, 1, 2, 2), dtype=torch.float32, device=DEVICE)}


def main():
    calibrator = Calibrator(data_loader=calib_data())

    engine = engine_from_network(
        network_from_onnx_path("identity.onnx"),
        config=CreateConfig(int8=True, calibrator=calibrator),
    )

    with TrtRunner(engine) as runner:
        inp_data = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=DEVICE)

        # 注意：运行器拥有输出缓冲区，可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果您想存储多个推理的结果，应该使用 `copy.deepcopy()`。
        #
        # 当您在 feed_dict 中提供 PyTorch 张量时，运行器将尝试对输出使用
        # PyTorch 张量。具体来说：
        # - 如果 `infer()` 的 `copy_outputs_to_host` 参数设置为 `True`（默认值），
        #       它将返回 CPU 内存中的 PyTorch 张量。
        # - 如果 `copy_outputs_to_host` 为 `False`，它将返回：
        #       - 如果您有支持 GPU 的 PyTorch 安装，则返回 GPU 内存中的 PyTorch 张量。
        #       - 否则返回 Polygraphy `DeviceView`。
        #
        outputs = runner.infer({"x": inp_data})

        # `copy_outputs_to_host` 默认为 True，所以输出应该是 CPU 内存中的
        # PyTorch 张量。
        assert isinstance(outputs["y"], torch.Tensor)
        assert outputs["y"].device.type == "cpu"

        assert torch.equal(outputs["y"], inp_data.to("cpu"))  # 这是一个恒等模型！


if __name__ == "__main__":
    main()
