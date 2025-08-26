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
该脚本使用 Polygraphy Runner API 在简单数据集上验证恒等模型的输出。
"""
import numpy as np
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner

# 假设这是一个非常大的数据集。
REAL_DATASET = [
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
]  # 绝对是真实数据

# 对于恒等网络，黄金输出值与输入值相同。
# 虽然这样的网络乍看之下似乎无用，但在某些情况下（比如这里！）非常有用。
EXPECTED_OUTPUTS = REAL_DATASET


def main():
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    with TrtRunner(build_engine) as runner:
        for data, golden in zip(REAL_DATASET, EXPECTED_OUTPUTS):
            # 注意：运行器拥有输出缓冲区，可以在 `infer()` 调用之间自由重用它们。
            #   因此，如果您想存储多个推理的结果，应该使用 `copy.deepcopy()`。
            outputs = runner.infer(feed_dict={"x": data})

            assert np.array_equal(outputs["y"], golden)

        print("验证成功！")


if __name__ == "__main__":
    main()
