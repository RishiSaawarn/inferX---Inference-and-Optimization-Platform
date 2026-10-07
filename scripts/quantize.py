import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_model(input_model_path: str, output_model_path: str):
    logger.info(f"Quantizing {input_model_path} to {output_model_path}...")
    try:
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8,
            # P2-16: add shape inference / prep before quantizing if needed in future
        )
        logger.info("Quantization complete.")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    quantize_model(args.input, args.output)
