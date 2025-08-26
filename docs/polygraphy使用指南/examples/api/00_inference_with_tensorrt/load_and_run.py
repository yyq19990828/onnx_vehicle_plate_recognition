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
这个脚本加载由 `build_and_run.py` 构建的 TensorRT 引擎并运行推理。
"""
import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner


def main():
    # 就像我们构建时一样，我们可以将多个加载器组合在一起
    # 以实现我们想要的行为。具体来说，我们想从文件中加载一个序列化的
    # 引擎，然后将其反序列化为 TensorRT 引擎。
    load_engine = EngineFromBytes(BytesFromPath("identity.engine"))

    # 推理过程与之前几乎完全相同：
    with TrtRunner(load_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # 注意: 运行器拥有输出缓冲区，并可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果要存储多次推理的结果，应使用 `copy.deepcopy()`。
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # 这是一个 identity 模型！

        print("推理成功！")


if __name__ == "__main__":
    main()
