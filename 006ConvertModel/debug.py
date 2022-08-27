import sys
sys.path.append("../utils")
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from IPython.display import Markdown, display
from fastseg import MobileV3Large
from openvino.runtime import Core

from notebook_utils import CityScapesSegmentation, segmentation_map_to_image, viz_result_image


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024 if IMAGE_WIDTH == 2048 else 512
DIRECTORY_NAME = "model"
BASE_MODEL_NAME = DIRECTORY_NAME + f"/fastseg{IMAGE_WIDTH}"

model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")

print("Downloading the Fastseg model ( if it has not been downloaded before)")
model = MobileV3Large.from_pretrained().cpu().eval()
print("Loaded PyTorch Fastseg model")

# Save the model
model_path.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), str(model_path))
print(f"Model saved at {model_path}")

if not onnx_path.exists():
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    print(dummy_input.shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=12,
        do_constant_folding=True,
    )

    print(f"ONNX model exported to {onnx_path}")
else:
    print(f"ONNX model {onnx_path} already exists")
