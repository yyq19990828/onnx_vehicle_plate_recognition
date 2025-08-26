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
来加载由 `build_and_run.py` 构建的 TensorRT 引擎并运行推理。
"""
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import TrtRunner, engine_from_bytes


def main():
    engine = engine_from_bytes(bytes_from_path("identity.engine"))

    with TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # 注意: 运行器拥有输出缓冲区，并可以在 `infer()` 调用之间自由重用它们。
        # 因此，如果要存储多次推理的结果，应使用 `copy.deepcopy()`。
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["output"], inp_data)  # 这是一个 identity 模型！

        print("推理成功！")


if __name__ == "__main__":
    main()
