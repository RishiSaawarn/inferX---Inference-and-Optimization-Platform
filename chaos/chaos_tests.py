import logging
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chaos")

BASE_URL = "http://localhost:8000"


def test_slow_backend():
    logger.info("Injecting slow backend failure...")
    try:
        response = requests.post(
            f"{BASE_URL}/chaos/inject",
            json={"type": "delay", "ms": 500, "model": "mobilenet_v3_small"},
        )
        if response.status_code == 200:
            logger.info("Successfully injected delay.")
            start = time.time()
            res = requests.post(
                f"{BASE_URL}/v1/inference", json={"model": "mobilenet_v3_small"}
            )
            end = time.time()
            logger.info(
                f"Request took {(end-start)*1000} ms. Status: {res.status_code}"
            )
        else:
            logger.warning("Chaos injection endpoint not available.")
    except Exception as e:
        logger.error(f"Failed to connect to API for chaos testing: {e}")


if __name__ == "__main__":
    test_slow_backend()
