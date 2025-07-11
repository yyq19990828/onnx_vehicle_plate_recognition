import cv2
import numpy as np

def preprocess_image(image: np.ndarray, input_shape: tuple = (640, 640)) -> tuple:
    """
    Preprocesses an image for ONNX model inference.

    Args:
        image (np.ndarray): The input image in BGR format.
        input_shape (tuple): The target shape (height, width) for the model.

    Returns:
        tuple: A tuple containing the preprocessed image tensor,
               the scaling factor, and the original image shape.
    """
    original_shape = image.shape[:2]  # (height, width)
    
    # Calculate scaling factor
    h, w = original_shape
    scale = min(input_shape[0] / h, input_shape[1] / w)
    
    # Resize the image with aspect ratio preservation
    unpad_w, unpad_h = int(round(w * scale)), int(round(h * scale))
    resized_image = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    # Create a new image with padding
    padded_image = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
    padded_image[:unpad_h, :unpad_w, :] = resized_image
    
    # Convert image to float32 and normalize to [0, 1]
    normalized_image = padded_image.astype(np.float32) / 255.0

    # Transpose the image from HWC to CHW format
    transposed_image = np.transpose(normalized_image, (2, 0, 1))

    # Add a batch dimension
    input_tensor = np.expand_dims(transposed_image, axis=0)

    return input_tensor, scale, original_shape