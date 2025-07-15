import logging
import threading
import onnxruntime

# Use a lock to ensure thread-safety for the one-time initialization
_preload_lock = threading.Lock()
_preload_done = False

def preload_onnx_libraries():
    """
    Preloads ONNX Runtime provider libraries, ensuring this action is performed only once.
    This is necessary to resolve "DLL not found" issues when using the GPU package,
    as the OS dynamic linker doesn't automatically search Python's site-packages.
    """
    global _preload_done
    # Fast check without locking
    if _preload_done:
        return

    with _preload_lock:
        # Double-check after acquiring the lock
        if _preload_done:
            return

        try:
            available_providers = onnxruntime.get_available_providers()
            logging.debug(f"Available ONNX Runtime providers: {available_providers}")

            if 'CUDAExecutionProvider' in available_providers:
                logging.info("Attempting to preload ONNX Runtime CUDA provider libraries...")
                onnxruntime.preload_dlls()
                logging.info("Successfully preloaded ONNX Runtime CUDA provider libraries.")
            else:
                logging.info("CUDAExecutionProvider not found. Skipping DLL preloading.")

        except ImportError:
            logging.warning("onnxruntime package not found, skipping preload.")
        except Exception as e:
            logging.error(f"An error occurred during ONNX Runtime DLL preloading: {e}", exc_info=True)
        finally:
            # Mark as done even if it fails, to avoid retrying.
            _preload_done = True
            
            
def get_best_available_providers(model_path: str) -> list[str]:
    """
    Returns the best available ONNX Runtime execution providers, prioritizing CUDA if it's functional.
    It performs a functional check for CUDA by attempting to load the specified model.
    
    Args:
        model_path (str): Path to the ONNX model to use for the functional check.

    Returns:
        list[str]: A list of the best available providers.
    """
    available_providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' not in available_providers:
        logging.info("CUDAExecutionProvider not available in this build of ONNX Runtime. Using CPU.")
        return ['CPUExecutionProvider']

    # Attempt to create a session with CUDA to confirm it's functional
    try:
        onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        logging.info("CUDAExecutionProvider is functional. Using CUDA and CPU providers.")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    except Exception as e:
        logging.warning(
            f"CUDAExecutionProvider is available but failed to initialize with model '{model_path}'. "
            f"This may be due to a driver mismatch or unsupported hardware. "
            f"Falling back to CPUExecutionProvider. Error: {e}"
        )
        return ['CPUExecutionProvider']