import logging
import threading
import onnxruntime
import json
import numpy as np
from typing import Optional, Dict, Any

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


def get_onnx_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """
    从ONNX模型中读取metadata信息
    
    Args:
        model_path (str): ONNX模型文件路径
        
    Returns:
        Optional[Dict[str, Any]]: metadata字典，如果没有则返回None
    """
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta()
        
        # 获取自定义metadata
        custom_metadata = {}
        if hasattr(metadata, 'custom_metadata_map'):
            custom_metadata = metadata.custom_metadata_map
        
        # 尝试解析names字段
        names_data = None
        if 'names' in custom_metadata:
            try:
                # 首先尝试JSON解析
                names_data = json.loads(custom_metadata['names'])
            except (json.JSONDecodeError, TypeError):
                try:
                    # 如果JSON解析失败，尝试Python字典的eval解析（Ultralytics格式）
                    names_data = eval(custom_metadata['names'])
                except Exception:
                    # 如果都失败了，直接返回字符串
                    names_data = custom_metadata['names']
        
        result = {
            'producer_name': getattr(metadata, 'producer_name', ''),
            'version': getattr(metadata, 'version', ''),
            'description': getattr(metadata, 'description', ''),
            'custom_metadata': custom_metadata,
            'names': names_data
        }
        
        return result
    except Exception as e:
        logging.warning(f"无法读取ONNX模型metadata: {e}")
        return None


def get_class_names_from_onnx(model_path: str) -> Optional[Dict[int, str]]:
    """
    从ONNX模型metadata中获取类别名称
    
    Args:
        model_path (str): ONNX模型文件路径
        
    Returns:
        Optional[Dict[int, str]]: 类别名称字典，key为类别ID，value为类别名称
    """
    metadata = get_onnx_metadata(model_path)
    if not metadata or not metadata.get('names'):
        return None
    
    names = metadata['names']
    
    # 处理不同的names格式
    if isinstance(names, dict):
        # 如果names是字典格式，尝试转换key为int
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (ValueError, TypeError):
            return {i: str(v) for i, v in enumerate(names.values())}
    elif isinstance(names, list):
        # 如果names是列表格式
        return {i: str(name) for i, name in enumerate(names)}
    
    return None


def get_model_info(model_path: str, default_input_shape: tuple = (640, 640)) -> dict:
    """
    统一获取ONNX模型的所有信息，只创建一次session
    
    Args:
        model_path (str): ONNX模型文件路径
        default_input_shape (tuple): 默认输入形状
        
    Returns:
        dict: 包含模型所有信息的字典
        {
            'session': onnxruntime.InferenceSession,
            'input_name': str,
            'output_names': List[str],
            'input_shape': tuple,
            'class_names': Optional[Dict[int, str]],
            'metadata': Optional[Dict[str, Any]]
        }
    """
    try:
        # 预加载库
        preload_onnx_libraries()
        
        # 创建session
        providers = get_best_available_providers(model_path)
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出信息
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        # 获取输入形状
        model_input_shape = session.get_inputs()[0].shape
        if (len(model_input_shape) >= 4 and 
            isinstance(model_input_shape[2], int) and model_input_shape[2] > 0 and
            isinstance(model_input_shape[3], int) and model_input_shape[3] > 0):
            input_shape = (model_input_shape[2], model_input_shape[3])
            logging.info(f"从ONNX模型读取到固定输入形状: {input_shape}")
        else:
            input_shape = default_input_shape
            logging.info(f"模型输入形状为动态 {model_input_shape}，使用默认形状: {input_shape}")
        
        # 获取metadata信息
        metadata = get_onnx_metadata(model_path)
        
        # 获取类别名称
        class_names = None
        if metadata and metadata.get('names'):
            names = metadata['names']
            # 处理不同的names格式
            if isinstance(names, dict):
                try:
                    class_names = {int(k): str(v) for k, v in names.items()}
                except (ValueError, TypeError):
                    class_names = {i: str(v) for i, v in enumerate(names.values())}
            elif isinstance(names, list):
                class_names = {i: str(name) for i, name in enumerate(names)}
        
        # 验证模型输出
        # 检查模型期望的batch维度
        model_input_shape = session.get_inputs()[0].shape
        expected_batch_size = model_input_shape[0] if isinstance(model_input_shape[0], int) and model_input_shape[0] > 0 else 1
        
        dummy_input = np.random.randn(expected_batch_size, 3, input_shape[0], input_shape[1]).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        output_shape = outputs[0].shape
        logging.info(f"模型输出形状: {output_shape}")
        logging.info(f"使用batch大小: {expected_batch_size}")
        
        return {
            'session': session,
            'input_name': input_name,
            'output_names': output_names,
            'input_shape': input_shape,
            'class_names': class_names,
            'metadata': metadata,
            'output_shape': output_shape
        }
        
    except Exception as e:
        logging.error(f"获取模型信息失败: {e}")
        return None