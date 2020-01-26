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
from models.pretrained.BigGAN.biggan import BigGAN128

from models.resnet18 import ResNet18
from models.densenet201 import DenseNet201

# Make sure to add to this as you write models.
model_dict = {
                'deepdecodernet': DeepDecoderNet,
                'stylegan': StyleGenerator,
                'biggan': BigGAN512,
             }

invert_model_dict = {
                        'resnet18': ResNet18,
                        'densenet201': DenseNet201,
                    }
