import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def postprocess_output(raw_output: np.ndarray, top_k: int = 5) -> list[dict]:
    """Applies softmax and gets top-k labels/probabilities"""
    if hasattr(raw_output, "detach"):
        raw_output = raw_output.detach().cpu().numpy()
        
    # Handle batch dimension if present
    if len(raw_output.shape) > 1 and raw_output.shape[0] == 1:
        raw_output = raw_output[0]
        
    probs = softmax(raw_output)
    
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "class_id": int(idx),
            "score": float(probs[idx])
        })
        
    return results
