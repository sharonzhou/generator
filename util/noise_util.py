import torch
import random
import numpy as np

def get_noise(args):
    if 'BigGAN' in args.model:
        from torch_pretrained_biggan import truncated_noise_sample
        noise = truncated_noise_sample(truncation=args.truncation, batch_size=args.batch_size)

    elif 'WGAN-GP' in args.model:
        noise = torch.randn(args.batch_size, 128)

    elif 'BEGAN' in args.model:
        noise = np.random.uniform(-1, 1, size=(args.batch_size, 64))
        noise = torch.FloatTensor(noise)
    
    else:
        raise Exception(f'{args.model} not found')

    return noise
