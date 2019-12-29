import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path

import torch
import torch.utils.data as data

import util

class ZTestDataset(data.Dataset):
    def __init__(self, args):
        
        # Target dir is the z_test dir
        self.target_dir = Path(args.z_test_target_dir)
        self.target_csv_path = self.target_dir / 'filenames.csv' 
        self.target_df = self.load_df(self.target_csv_path)
        
        self.mask_dir = Path(args.mask_dir)
        self.masks = self.get_mask_names(self.target_df)

        self.targets = self.get_image_names(self.target_df)
        self.target_shape = self.get_target_shape(self.targets)
        self.num_targets = len(self.targets)

        # Create z-test vecs
        # For deep decoder net input, reshape vectors
        if args.model == 'DeepDecoderNet' and args.use_custom_input_noise:
            self.z_tests = torch.cat([util.get_deep_decoder_input_noise(self.target_shape)
                                      for _ in range(self.num_targets)],
                                      dim=0)
        else:
            self.z_tests = torch.stack([util.get_input_noise()
                                        for _ in range(self.num_targets)],
                                        dim=0)
       
    def load_df(self, csv_path):
        return pd.read_csv(csv_path)

    def get_image_names(self, df):
        return df['filename'].values

    def get_mask_names(self, df):
        return df['mask'].values
    
    def get_target_shape(self, target_names):
        target_name = target_names[0]
        target = util.get_image(self.target_dir,
                                target_name)
        return target.shape

    def __getitem__(self, index):

        # Get z_test noise
        z_test = self.z_tests[index]
        
        # Get target image
        target_name = self.targets[index]
        target = util.get_image(self.target_dir,
                                target_name)

        # Mask image
        mask_name = self.masks[index]
        mask = util.get_image(self.mask_dir,
                              mask_name)
        
        return z_test, target, mask

    def __len__(self):
        return self.num_targets
