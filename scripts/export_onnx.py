import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_onnx")


def get_model(model_name: str):
    if model_name == "mobilenet_v3_small":
        return models.mobilenet_v3_small(pretrained=True)
    elif model_name == "resnet18":
        return models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def export_to_onnx(model_name: str, output_path: str):
    logger.info(f"Loading {model_name}...")
    model = get_model(model_name)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    logger.info(f"Exporting to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info("Export completed.")

    # Numerical validation
    logger.info("Validating numerical equivalence...")
    with torch.no_grad():
        pytorch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_out = session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pytorch_out - onnx_out))
    mean_diff = np.mean(np.abs(pytorch_out - onnx_out))
    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    if max_diff > 1e-3:
        logger.warning("Numerical divergence exceeds 1e-3!")
        return False
    logger.info("Numerical validation passed.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g. mobilenet_v3_small)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output ONNX file path"
    )
    args = parser.parse_args()

    success = export_to_onnx(args.model, args.output)
    if not success:
        exit(1)
