import sys
import torch
import torchvision.models as models
import onnx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_model(model_name: str, output_path: str):
    logger.info(f"Exporting {model_name}...")
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        logger.error(f"Unknown model: {model_name}")
        sys.exit(1)
        
    model.eval()
    
    # Needs a dynamic batch axis for the batch scheduler
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,  # P2-16: Update opset to modern version
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Successfully exported {model_name} to {output_path}")
        
        # P2-16: Add ONNX checker
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX validation passed.")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    export_model(args.model, args.output)
