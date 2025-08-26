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
这个脚本使用 ONNX-Runtime 和 TensorRT 运行一个 identity 模型，
然后比较输出。
"""
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc


def main():
    # OnnxrtRunner 需要一个 ONNX-RT 会话。
    # 我们可以使用 SessionFromOnnx 延迟加载器轻松构建一个：
    build_onnxrt_session = SessionFromOnnx("identity.onnx")

    # TrtRunner 需要一个 TensorRT 引擎。
    # 要从 ONNX 模型创建一个，我们可以将几个延迟加载器链接在一起：
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    # `Comparator.run()` 将使用综合输入数据分别运行每个运行器，并
    #   返回一个 `RunResults` 实例。有关详细信息，请参阅 `polygraphy/comparator/struct.py`。
    #
    # 提示：要使用自定义输入数据，您可以将 `Comparator.run()` 中的 `data_loader` 参数设置
    #   为一个生成器或可迭代对象，该对象产生 `Dict[str, np.ndarray]`。
    run_results = Comparator.run(runners)

    # `Comparator.compare_accuracy()` 检查运行器之间的输出是否匹配。
    #
    # 提示：`compare_func` 参数可用于控制如何比较输出（有关详细信息，请参阅 API 参考）。
    #   默认的比较函数由 `CompareFunc.simple()` 创建，但如果我们想更改
    #   默认参数（例如容差），我们可以显式地构造它。
    assert bool(
        Comparator.compare_accuracy(
            run_results, compare_func=CompareFunc.simple(atol=1e-8)
        )
    )

    # 使用距离度量比较进行更全面的评估
    assert bool(
        Comparator.compare_accuracy(
            run_results,
            compare_func=CompareFunc.distance_metrics(
                l2_tolerance=1e-5,                    # 允许的最大 L2 范数（欧几里得距离）
                cosine_similarity_threshold=0.99,     # 最小余弦相似度（角度相似度）
            )
        )
    )
    print("所有输出都使用距离度量（L2 范数，余弦相似度）匹配")
    
    # 使用质量度量进行信号质量评估
    assert bool(
        Comparator.compare_accuracy(
            run_results,
            compare_func=CompareFunc.quality_metrics(
                psnr_tolerance=50.0,                  # 最小峰值信噪比（dB）
                snr_tolerance=25.0                    # 最小信噪比（dB）
            )
        )
    )
    print("所有输出都使用质量度量（PSNR，SNR）匹配")

    # 我们可以使用 `RunResults.save()` 方法将推理结果保存到 JSON 文件中。
    # 如果您想单独生成和比较结果，这可能很有用。
    run_results.save("inference_results.json")


if __name__ == "__main__":
    main()
