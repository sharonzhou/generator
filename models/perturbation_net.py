"""
Perturbation Net class that takes in an existing model G (w/ frozen parameters - or freezes them here too) and adds trainable layers that injects noise tto the finer layers, or the last four scales of convolutions (higher level layers h=n-5).
# TODO: create perturbation network R - make it copy G fine layer shapes - need it to do forward pass during model.forward
# so first do model.forward of the usual model G on the high level layers, then replace with R? or interchange layers btw G and R and wrap the full model in a larger thing - where the G layers are frozen, but the R layers are not - I wonder if there's an easier way to do this so it's easy to retrofit any G - maybe if R copies all of G_finelayers, freezes itself on those params, then creates its own around each that's not frozen, that could work and the second forward pass is just R


"""

import torch.nn as nn
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict

import util
from models.base_net import BaseNet

class PerturbationNet(BaseNet):
    def __init__(self, inner_model, **kwargs):
        super(PerturbationNet, self).__init__()

        # Find the 5 areas to stop and include perturbed layer
        # Determine the sizes of each of them
        # Initialize self.perturb1-5 layers

        self.model0a = nn.Sequential(*list(inner_model.generator.layers[:-1]))
        self.model0b = nn.Sequential(*[inner_model.generator.layers._modules['14'].bn_0, 
                                      inner_model.generator.layers._modules['14'].conv_0,
                                      ])
        out_channels = inner_model.generator.layers._modules['14'].conv_0.out_channels
        self.perturb0 = nn.Linear(out_channels, out_channels, bias=False)

        self.model1 = nn.Sequential(*[inner_model.generator.layers._modules['14'].bn_1, 
                                      inner_model.generator.layers._modules['14'].conv_1,
                                     ])
        out_channels = inner_model.generator.layers._modules['14'].conv_1.out_channels
        self.perturb1 = nn.Linear(out_channels, out_channels, bias=False)
        
        self.model2 = nn.Sequential(*[inner_model.generator.layers._modules['14'].bn_2, 
                                      inner_model.generator.layers._modules['14'].conv_2,
                                     ])
        out_channels = inner_model.generator.layers._modules['14'].conv_2.out_channels
        self.perturb2 = nn.Linear(out_channels, out_channels, bias=False)
        
        self.model3 = nn.Sequential(*[inner_model.generator.layers._modules['14'].bn_3, 
                                      inner_model.generator.layers._modules['14'].conv_3,
                                     ])
        out_channels = inner_model.generator.layers._modules['14'].conv_3.out_channels
        self.perturb3 = nn.Linear(out_channels, out_channels, bias=False)
        
        self.model4a = nn.Sequential(*[inner_model.generator.layers._modules['14'].relu, 
                                      ])
        self.model4b = nn.Sequential(*list(inner_model.generator.children())[-4:-1])
        out_channels = inner_model.generator.conv_to_rgb.out_channels
        self.perturb4 = nn.Linear(out_channels, out_channels, bias=False)

        self.model5 = nn.Sequential(*list(inner_model.generator.children())[-1:])

    
    def forward(self, z, class_label, truncation):
        assert 0 < truncation <= 1
        
        embed = self.embeddings(class_label)
        cond_vector = torch.cat((z, embed), dim=1)

        out = self.model0a(cond_vector, truncation)
        out = self.model0b(out)
        #out = self.perturb0(out)
        
        out = self.model1(out, truncation, cond_vector)
        #out = self.perturb1(out)
        
        out = self.model2(out, truncation, cond_vector)
        #out = self.perturb2(out)
        
        out = self.model3(out, truncation, cond_vector)
        #out = self.perturb3(out)
        
        out = self.model4a(out)
        out = self.model4b(out)
        #out = self.perturb4(out)
        
        out = out[:, :3, ...]
        out = self.model5(out)

        return out

    def args_dict(self):
        model_args = {}
        return model_args
