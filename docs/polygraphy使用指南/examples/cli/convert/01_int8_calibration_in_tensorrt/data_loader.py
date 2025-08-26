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
定义一个 `load_data` 函数，该函数返回一个生成器，该生成器产生
feed_dicts，以便此脚本可以用作
--data-loader-script 命令行参数的参数。
"""
import numpy as np

INPUT_SHAPE = (1, 1, 2, 2)


def load_data():
    for _ in range(5):
        yield {
            "x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)
        }  # 仍然是完全真实的数据
