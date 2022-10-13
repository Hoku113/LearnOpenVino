import collections
import os
import sys
import time

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core
from decoder import OpenPoseDecoder

from function.function import *

base_model_dir = "model"
model_name = "human-pose-estimation-0001"
precision = "FP16-INT8"
model_path = f"model/intel/{model_name}/{precision}/{model_name}.xml"
model_weights_path = f"model/intel/{model_name}/{precision}/{model_name}.bin"

ie_core = Core()

model = ie_core.read_model(model_path, model_weights_path)
compiled_model = ie_core.compile_model(model, device_name="CPU")

#  Get the input and output anmes of nodes.
input_layer = compiled_model.input(0)
output_layers = list(compiled_model.outputs)

# debug
print(input_layer)
print(output_layers)

# get the input size
height, width = list(input_layer.shape)[2:]

decoder = OpenPoseDecoder()

run_pose_estimation(compiled_model, width, height, decoder, source=0, flip=False, use_popup=False)