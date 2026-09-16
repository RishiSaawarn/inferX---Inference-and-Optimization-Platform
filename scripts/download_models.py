import torchvision.models as models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

def download_models():
    logger.info("Downloading MobileNetV3-Small weights...")
    models.mobilenet_v3_small(pretrained=True)
    logger.info("Downloading ResNet18 weights...")
    models.resnet18(pretrained=True)
    logger.info("Models downloaded successfully.")

if __name__ == "__main__":
    download_models()
