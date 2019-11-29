import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path

import torch
import torch.utils.data as data

import util

class Dataset(data.Dataset):
    def __init__(self, args):
        
        self.target_dir = Path(args.target_dir)
        self.target_csv_path = self.target_dir / 'filenames.csv' 
        self.target_df = self.load_df(self.target_csv_path)
        
        self.masks = self.get_mask_names(self.target_df)

        self.targets = self.get_image_names(self.target_df)
        self.target_shape = self.get_target_shape(self.targets)
        self.num_targets = len(self.targets)
       
        # Get z-test targets
        self.z_test_target_dir = Path(args.z_test_target_dir)
        self.z_test_target_csv_path = self.z_test_target_dir / 'filenames.csv' 
        self.z_test_target_df = self.load_df(self.z_test_target_csv_path)
        
        self.z_test_targets = self.get_image_names(self.z_test_target_df)
        self.num_z_test = len(self.z_test_targets)

        # Create input noise and z-test vecs
        # For deep decoder net input, reshape vectors
        if args.model == 'DeepDecoderNet' and args.use_custom_input_noise:
            self.input_noises = torch.cat([util.get_deep_decoder_input_noise(self.target_shape)
                                           for _ in range(self.num_targets)],
                                           dim=0)
            self.z_tests = torch.cat([util.get_deep_decoder_input_noise(self.target_shape)
                                      for _ in range(self.num_z_test)],
                                      dim=0)
        else:
            self.input_noises = torch.cat([util.get_input_noise()
                                           for _ in range(self.num_targets)],
                                           dim=0)
            self.z_tests = torch.cat([util.get_input_noise()
                                      for _ in range(self.num_z_test)],
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

        # Get input noise
        input_noise = self.input_noises[index]

        # Get target image
        target_name = self.targets[index]
        target = util.get_image(self.target_dir,
                                target_name)

        # Mask image
        mask_name = self.masks[index]
        mask = util.get_image(self.mask_dir,
                              mask_name)

        # Get z-test vec and targets
        z_test_index = self.num_z_test % self.num_targets
        z_test = self.z_tests[z_test_index]
        z_test_target_name = self.z_test_targets[z_test_index]
        z_test_target = util.get_image(self.z_test_target_dir,
                                       z_test_target_name)

        return input_noise, target, mask, z_test_target, z_test

    def __len__(self):
        return self.num_targets
