import argparse
import getpass
import json
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import util


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='generator_evaluation')
        
        self.parser.add_argument('--name', type=str, required=True,
                                 help='Name of run')

        # Model args
        self.parser.add_argument('--model', type=str, choices=('DeepDecoderNet'), default='DeepDecoderNet',
                                 help='Model to use. Basic conv1x1 from the Deep Decoder paper is currently the only model available')

        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='Batch size.')

        self.parser.add_argument('--ckpt_path', type=str, default='',
                                 help='Path to checkpoint to load. If empty, start from scratch.')

        self.parser.add_argument('--image_name', type=str, default='lena.png',
                                 help='Image name to train on (to compress).')

        self.parser.add_argument('--mask_name', type=str, default=None,
                                 help='Mask name to avoid backproping gradients through.')
        
        self.parser.add_argument('--data_dir', type=str, default='images',
                                 help='Path to image(s) to run through generator.')

        self.parser.add_argument('--mask_dir', type=str, default='masks',
                                 help='Path to mask(s) to run through generator.')
        
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                                 help='Comma-separated list of GPU IDs. Use -1 for CPU.')

        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'),
                                 help='Initialization method to use for conv kernels and linear weights.')

        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for the DataLoader.')
        
        self.parser.add_argument('--save_dir', type=str, default='ckpts/',
                                 help='Directory in which to save model checkpoints.')
       
        self.parser.add_argument('--toy', action='store_true', help='Use small dataset or not.')

        self.parser.add_argument('--num_visuals', type=int, default=4, help='Number of visuals to display per eval.')

        # Custom parameters for model input and model architecture depending on input image
        self.parser.add_argument('--use_custom_input_noise', action='store_true', help='Use custom noise -- relevant only for Deep Decoder Net right now.')
        self.parser.add_argument('--disable_batch_norm', action='store_true', help='Disable batch norm if on (good for hexes to disable, but may be better to keep it for other images).')
    
    def parse_args(self):
        args = self.parser.parse_args()

        # Save args to a JSON file
        save_dir = os.path.join(args.save_dir, '{}_{}'.format(getpass.getuser(), args.name))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir = save_dir

        args.start_epoch = 1  # Gets updated if we load a checkpoint
        args.is_training = self.is_training

        # Set up available GPUs
        args.gpu_ids = util.args_to_list(args.gpu_ids, allow_empty=True, arg_type=int, allow_negative=False)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        return args
