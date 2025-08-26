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
创建一个 TensorRT 构建器配置并启用 FP16 策略。
"""
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig


# 如果我们定义一个名为 `load_config` 的函数，polygraphy 可以使用它来
# 创建构建器配置。
#
# 提示：如果我们的函数不叫 `load_config`，我们可以用脚本参数显式指定
# 名称，用冒号分隔。例如：`create_config.py:my_func`。
@func.extend(CreateConfig())
def load_config(config):
    # 注意：func.extend() 使此函数的签名为 `(builder, network) -> config`
    # 有关其工作原理的详细信息，请参阅 examples/api/03_interoperating_with_tensorrt

    config.set_flag(trt.BuilderFlag.FP16)

    # 注意，我们不需要返回任何东西 - `extend()` 会为我们处理！
