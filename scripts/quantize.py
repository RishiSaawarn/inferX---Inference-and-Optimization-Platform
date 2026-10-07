import argparse
import logging
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantize")


def quantize_model(input_path: str, output_path: str):
    logger.info(f"Quantizing {input_path} to INT8...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Using dynamic quantization for simplicity, as it's a good baseline.
    quantize_dynamic(
        model_input=input_path, model_output=output_path, weight_type=QuantType.QInt8
    )
    logger.info(f"Quantization complete. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input ONNX file path")
    parser.add_argument(
        "--output", type=str, required=True, help="Output Quantized ONNX file path"
    )
    args = parser.parse_args()
    quantize_model(args.input, args.output)
