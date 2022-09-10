import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from addict import Dict
from openvino.tools.pot.api import DataLoader, Metric
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.runtime import Core
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Set the data and model directories
DATA_DIR = 'data'
MODEL_DIR = 'model'

try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
except OSError as e:
    print(e)

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_0",  pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)

onnx_model_path = Path(MODEL_DIR)/ 'mobilenet_v2.onnx'
ir_model_xml = onnx_model_path.with_suffix('.xml')
ir_model_bin = onnx_model_path.with_suffix('.bin')

torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)

# Run OpenVINO Model Optimization tool to convert ONNX to OpenVINO IR
!mo --framework=onnx --data_type=FP16 --input_shape[1,chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_0 3, 32, 32] -m $onnx_model_path --output_dir $MODEL_DIR