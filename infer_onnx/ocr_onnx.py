import onnxruntime
import numpy as np
import logging

from .utils import preload_onnx_libraries, get_best_available_providers

class ColorLayerONNX:
    """
    颜色与层数ONNX推理类
    """
    def __init__(self, model_path):
        # Ensure ONNX Runtime libraries are preloaded if necessary
        preload_onnx_libraries()
        
        providers = get_best_available_providers(model_path)
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, img):
        """
        输入: 预处理后的图像 (np.ndarray, shape: [1, 3, H, W])
        返回: (color_logits, layer_logits)
        """
        outputs = self.session.run(None, {self.input_name: img})
        # 假设模型输出为 [color_logits, layer_logits]
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        # 若模型输出合并，则需拆分
        return outputs[0], outputs[1] if len(outputs) > 1 else (outputs[0], None)

class OCRONNX:
    """
    OCR字符识别ONNX推理类
    """
    def __init__(self, model_path):
        # Ensure ONNX Runtime libraries are preloaded if necessary
        preload_onnx_libraries()

        providers = get_best_available_providers(model_path)
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, img):
        """
        输入: 预处理后的图像 (np.ndarray, shape: [1, 3, 48, 168])
        返回: onnx输出 (通常为概率分布)
        """
        feed = {name: img for name in self.input_names}
        outputs = self.session.run(self.output_names, feed)
        return outputs