"""
train.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import util
import models
from args import TrainArgParser
from logger import TrainLogger
from saver import ModelSaver

def train(args):
    
    # Instantiate input and target: Input is noise tensor, target is image
    target_image = util.get_target_image(args)
    input_noise = util.get_input_noise(args)

    # Instantiate mask
    mask = util.get_mask(args)
    setattr(args, 'mask', mask)

    # For deep decoder net input, reshape input noise and do not parallelize
    if args.model == 'DeepDecoderNet':
        setattr(args, 'target_image_shape', target_image.shape)
       
        if args.use_custom_input_noise:
            input_noise = util.get_deep_decoder_input_noise(target_image.shape)
        else:
            # Do not parallelize (reshape noise has issues in fc layer), use 1st gpu
            gpu_ids = args.gpu_ids
            gpu_id = [gpu_ids[0]]
            setattr(args, 'gpu_ids', gpu_id)

    print(f'Input: {input_noise.shape}')
    print(f'Target: {target_image.shape}')
    print(f'Mask: {mask.shape}')

    # Load model
    model_fn = models.__dict__[args.model]
    model = model_fn(**vars(args))
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and loss
    parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    loss_fn = util.get_loss_fn(args.loss_fn, args)

    # Get logger and saver
    logger = TrainLogger(args)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric)
    
    print(f'Logs: {logger.log_dir}')
    print(f'Ckpts: {args.save_dir}')

    # Train model
    logger.log_hparams(args)
    while not logger.is_finished_training():
        logger.start_epoch()

        # Input is noise tensor, target is image
        with torch.set_grad_enabled(True):
            input_noise.to(args.device)
            logits = model.forward(input_noise)

            # Masked loss
            masked_loss = torch.zeros(1, requires_grad=True).to(args.device)
            
            if mask is None:
                masked_logits = logits
                masked_target_image = target_image
            else:
                # TODO: debug shapes here
                masked_logits = logits[mask] 
                masked_target_image = target_image[mask]
            masked_loss = loss_fn(masked_logits, masked_target_image.to(args.device))
           
            loss = torch.zeros(1).to(args.device)
            with torch.no_grad():
                loss = loss_fn(logits, target_image.to(args.device))

            logger.log_status(input_noise, masked_logits, logits, target_image, masked_loss, loss)
            
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

        # TODO: Create function for metrics
        metrics = {'masked_loss': masked_loss.item(), 'loss': loss.item()}
        saver.save(logger.epoch, model, optimizer, args.device, metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics)

if __name__ == "__main__":
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
