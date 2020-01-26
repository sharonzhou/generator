import torch
import torch.utils.data as data

from .dataset import Dataset
from .z_test_dataset import ZTestDataset
from .invert_dataset import InvertDataset

def get_loader(args, phase='train'):
    if phase == 'train':
        dataset = Dataset(args, phase)
    elif phase == 'invert':
        dataset = InvertDataset(args)
    else:
        dataset = ZTestDataset(args)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=phase=='train')
    return loader
