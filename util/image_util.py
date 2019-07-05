import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as interpolation
from skimage import morphology
from scipy import interpolate
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable


def get_image(img_dir, img_name):
    # Load data and image
    img_dir = Path(img_dir)
    img_path = img_dir / img_name
    img = Image.open(img_path).convert('RGB')

    # Transform image
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transform = transforms.Compose([t for t in transforms_list if t])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
   
    if torch.cuda.is_available():
        img = img.cuda()

    return img
   
def get_target_image(args):
    return get_image(args.data_dir, args.image_name)

def get_mask(args):
    if not args.mask_dir or not args.mask_name:
        mask = torch.ones(args.target_image_shape)
    else:
        # Load mask
        mask = get_image(args.mask_dir, args.mask_name)
        mask = mask.float()
    
    if torch.cuda.is_available():
        mask = mask.cuda()
  
    return mask

def get_input_noise(args, dist='gaussian', latent_dim=100):
    # Sample a fixed noise vector z \in R^latent_dim
    noise_tensor = Variable(torch.FloatTensor(latent_dim))

    if dist == 'uniform':
        # Sample with uniform distribution
        noise_tensor.data.uniform_(-1, 1)
    else:
        # Sample with spherical gaussian z \in R^latent_dim ~ N(0, I)
        noise_tensor.data.normal_(mean=0, std=1) 

    if torch.cuda.is_available():
        noise_tensor = noise_tensor.cuda()

    return noise_tensor

def get_deep_decoder_input_noise(image_shape, num_channels=128, num_up=5):
    # Create custom fixed noise input (stays same across epochs) for deep decoder net
    total_upsample = 2 ** num_up

    height = int(image_shape[2] / total_upsample)
    width = int(image_shape[3] / total_upsample)

    noise_shape = [1, num_channels, height, width]
    noise_tensor = Variable(torch.zeros(noise_shape))

    noise_tensor.data.uniform_() 
    noise_tensor /= 10.

    if torch.cuda.is_available():
        noise_tensor = noise_tensor.cuda()

    return noise_tensor

def _make_rgb(image):
    """Tile a NumPy array to make sure it has 3 channels."""
    if image.shape[-1] != 3:
        tiling_shape = [1] * (len(image.shape) - 1) + [3]
        return np.tile(image, tiling_shape)
    else:
        return image

def normalize_to_image(img):
    """Normalizes img to be in the range 0-255."""
    img *= 255.
    return img

def convert_image_from_tensor(image):
    # Remove batchsize
    image = np.squeeze(image, 0)

    # Move channel to last in shape
    image = np.moveaxis(image, 0, -1)

    # Normalize
    image = normalize_to_image(image)

    # Convert to floats
    image = image.astype(float)

    return image

def concat_images(images, spacing=10):
    """Concatenate a list of images to form a single row image.
    Args:
        images: Iterable of numpy arrays, each holding an image.
        Must have same height, num_channels, and have dtype np.uint8.
        spacing: Number of pixels between each image.
    Returns: Numpy array. Result of concatenating the images in images into a single row.
    """
    # Make array of all white pixels with enough space for all concatenated images

    row_index = 0
    col_index = 1
    channel_index = 2

    assert spacing >= 0, 'Invalid argument: spacing {} is not non-negative'.format(spacing)
    assert len(images) > 0, 'Invalid argument: images must be non-empty'
    num_rows = images[0].shape[row_index]
    num_channels = images[0].shape[channel_index]

    assert all([img.shape[row_index] == num_rows and img.shape[channel_index] == num_channels for img in images]),\
        'Invalid image shapes: images must have same num_channels and height'

    num_cols = sum([img.shape[col_index] for img in images]) + spacing * (len(images) - 1)
    concatenated_images = np.full((num_rows, num_cols, num_channels), fill_value=255, dtype=np.uint8)

    # Paste each image into position
    col = 0
    for img in images:
        num_cols = img.shape[col_index]
        concatenated_images[:, col:col + num_cols, :] = img
        col += num_cols + spacing

    return concatenated_images 

class UnNormalize(object):
    """Unnormalizes an image tensor"""
    def __init__(self, mean, std, inplace=True):
        self.mean = mean
        self.std = std
        self.inplace=inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.inplace:
            unnormed_tensor = tensor
        else:
            unnormed_tensor = torch.Tensor(tensor)

        for t, m, s in zip(unnormed_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
