import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path

import torch
import torch.utils.data as data

import util

class InvertDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.model = util.get_model(args)
        self.num_epochs = args.num_invert_epochs
       
    def __getitem__(self, index):
        # Sample z
        noise = util.get_noise(self.args)

        # Generate output - do this on GPU
        self.model = self.model.cuda()
        noise = noise.cuda()
        if 'BigGAN' in self.model_name:
            from torch_pretrained_biggan import one_hot_from_int
            class_vector = one_hot_from_int(207, batch_size=1) # TODO: check if batch size 1 makes sense for single getitem
            class_vector = torch.from_numpy(class_vector)
            class_vector = class_vector.cuda()

            image = self.model.forward(noise, class_vector, args.truncation).float() 
        else:
            image = self.model.forward(noise).float() 

        # Take off GPU
        self.model = self.model.cpu()
        noise = noise.cpu()

        # Normalize image
        image = (image + 1.) / 2.

        # Return pair
        return image, noise

    def __len__(self):
        return self.num_epochs
