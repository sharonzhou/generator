"""
model.py
	Defines PyTorch nn.Module classes.
	Each should implement a constructor __init__(self, config)
	and a forward pass forward(self, x)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch.nn as nn
from torchvision import models

from models.deep_decoder_net import DeepDecoderNet
from models.stylegan import StyleGenerator

# Make sure to add to this as you write models.
model_dict = {
                'deepdecodernet': DeepDecoderNet,
                'stylegan': StyleGenerator,
             }
