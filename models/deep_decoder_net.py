"""
Deep Decoder model
http://www.reinhardheckel.com/papers/deep_decoder.pdf
https://github.com/reinhardh/supplement_deep_decoder
"""

import torch.nn as nn
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import numpy as np

import util
from models.base_net import BaseNet

class Upsample(nn.Module):
    def __init__(self,  scale_factor, mode):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=True)

class DeepDecoderNet(BaseNet):
    def __init__(self,
                 target_image_shape,
                 mask=None,
                 num_channels=128,
                 num_up=5,
                 default_noise_size=100,
                 use_custom_input_noise=False,
                 disable_batch_norm=False,
                 use_intermediate_logits=False,
                 **kwargs):
        super(DeepDecoderNet, self).__init__()

        self.mask = mask
        self.target_image_shape = target_image_shape
        self.use_custom_input_noise = use_custom_input_noise
        self.default_noise_size = default_noise_size

        self.num_channels = num_channels # AKA. k
        self.num_up = num_up # AKA. num_channels_up
        self.num_output_channels = target_image_shape[1]

        if not self.use_custom_input_noise:
            # Added FC layer to account for standard noise vector
            num_output_height = target_image_shape[2]
            height_factor = int(np.log2(num_output_height))
            input_height_factor = height_factor - self.num_up
            
            self.input_height = 2 ** input_height_factor
            self.input_width = self.input_height
            self.fc = nn.Linear(self.default_noise_size, 
                                self.num_channels * self.input_height * self.input_width)
      
        # Layers in Deep Decoder Net
        # Initialize weights of conv layers to Kaiming normal
        layers = []
        for i in range(self.num_up):
            layers.append(Upsample(scale_factor=2, mode='bilinear'))

            conv = nn.Conv2d(self.num_channels, self.num_channels, 1)
            nn.init.kaiming_normal(conv.weight.data, nonlinearity='relu')
            layers.append(conv)      
            
            if not disable_batch_norm:
                layers.append(nn.BatchNorm2d(self.num_channels))

            layers.append(nn.ReLU(inplace=True))
        
        last_conv = nn.Conv2d(self.num_channels, self.num_output_channels, 1)
        nn.init.kaiming_normal(last_conv.weight.data, nonlinearity='relu')
        layers.append(last_conv)
        
        if not use_intermediate_logits:
            # Output probabilities directly
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        

    def forward(self, X):
        """
        Parameters
        ----------
        X : tensor
            Shape (batch_size x c x h x w)
        """
        # Include FC layer to reshape noise vector
        if not self.use_custom_input_noise:
            X = self.fc(X)
            X = X.view(1, self.num_channels, self.input_height, self.input_width)
        
        # Run Deep Decoder Net
        out = self.model(X)

        return out

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `DeepDecoderNet(**model_args)`.
        """
        model_args = {
                        'target_image_shape': self.target_image_shape,
                        'num_channels': self.num_channels,
                        'num_up': self.num_up,
                        'default_noise_size': self.default_noise_size,
                        'use_custom_input_noise': self.use_custom_input_noise,
                     }

        return model_args
