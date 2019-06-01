"""
Base Net class that models inherit from. Includes masking.
"""

import torch.nn as nn
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import numpy as np

import util

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.mask = None
        self.model = None

    def mask_hook(self, module, grad_input, grad_output):
        grad = grad_input[0]
        self.mask = self.mask.byte()
        grad[~self.mask] = 0
        return tuple([grad])

    def mask_pass(self):
        self.model[-1].register_backward_hook(self.mask_hook)
        return 

    def forward_pass(self, X):
        raise NotImplementedError('Subclass of BaseNet should implement forward pass.')

    def forward(self, X):
        raise NotImplementedError('Subclass of BaseNet should implement forward pass.')
      #  if self.mask is None:
      #      return self.forward_pass(X)
      #  else:
      #      out = self.forward_pass(X)
      #      self.mask_pass()
      #      return out

    def args_dict(self):
        model_args = {}
        return model_args
