import numpy as np
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)

try:
    from inferx_preprocess import preprocess_image
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    logger.info("C++ preprocessing module not available, using purely Python fallback.")

def decode_image_base64(b64_str: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def python_preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    
    # HWC to CHW
    img_arr = np.transpose(img_arr, (2, 0, 1))
    return np.expand_dims(img_arr, axis=0) # Add batch dim

def preprocess_request(input_data: str | list) -> np.ndarray:
    """Entry point for normalizing API standard input to Model-ready tensors"""
    if isinstance(input_data, list):
        return np.array(input_data, dtype=np.float32)
        
    # Assume base64 string
    img = decode_image_base64(input_data)
    
    if HAS_CPP:
        try:
            # Resize in Python, normalize in C++
            img = img.resize((224, 224))
            img_arr = np.array(img, dtype=np.uint8)
            chw = preprocess_image(img_arr)
            return np.expand_dims(chw, axis=0)
        except Exception as e:
            logger.warning(f"C++ preprocessing failed, falling back to Python: {e}")
            
    return python_preprocess(img)
