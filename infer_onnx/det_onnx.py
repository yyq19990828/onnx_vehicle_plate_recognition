import onnxruntime
import numpy as np
import logging
from typing import List, Tuple

from .utils import preload_onnx_libraries, get_best_available_providers
from utils.image_processing import preprocess_image
from utils.nms import non_max_suppression

class DetONNX:
    """
    A class for performing object detection using an ONNX model.
    """

    def __init__(self, onnx_path: str, input_shape: Tuple[int, int] = (640, 640), conf_thres: float = 0.5, iou_thres: float = 0.5):
        # Ensure ONNX Runtime libraries are preloaded if necessary
        preload_onnx_libraries()

        """
        Initializes the DetONNX instance.

        Args:
            onnx_path (str): The path to the ONNX model file.
            input_shape (Tuple[int, int]): The expected input shape (height, width) of the model.
            conf_thres (float): Confidence threshold for NMS.
            iou_thres (float): IoU threshold for NMS.
        """
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Create an ONNX Runtime session using the best available providers
        providers = get_best_available_providers(self.onnx_path)
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def __call__(self, image: np.ndarray) -> Tuple[List[np.ndarray], tuple]:
        """
        Performs inference on a single image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            Tuple[List[np.ndarray], tuple]: A tuple containing the list of detections and the original image shape.
                                            Each detection is an array of shape [num_detections, 6]
                                            representing (x1, y1, x2, y2, conf, class_id).
        """
        # Preprocess the image
        input_tensor, scale, original_shape = preprocess_image(image, self.input_shape)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # The primary output is usually the first one
        prediction = outputs[0]

        # The model's output format is [center_x, center_y, width, height, ...],
        # where coordinates are normalized.
        # We need to scale them to the input image size (e.g., 640x640).
        prediction[..., 0] *= self.input_shape[1]  # x_center
        prediction[..., 1] *= self.input_shape[0]  # y_center
        prediction[..., 2] *= self.input_shape[1]  # width
        prediction[..., 3] *= self.input_shape[0]  # height

        # Post-process the prediction with NMS
        detections = non_max_suppression(
            prediction,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres
        )

        # Scale boxes back to original image size
        if detections and len(detections[0]) > 0:
            detections[0][:, :4] /= scale

        return detections, original_shape