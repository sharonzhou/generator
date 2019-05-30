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

    # For deep decoder net input, reshape input noise and do not parallelize
    if args.model == 'DeepDecoderNet':
        setattr(args, 'target_image_shape', target_image.shape)
        
        # Do not parallelize (reshape noise has issues in fc layer), use 1st gpu
        gpu_ids = args.gpu_ids
        gpu_id = [gpu_ids[0]]
        setattr(args, 'gpu_ids', gpu_id)
        

    print(f'Input: {input_noise.shape}')
    print(f'Target: {target_image.shape}')

    print(f'Logs: {args.save_dir}')

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

    # Train model
    logger.log_hparams(args)
    while not logger.is_finished_training():
        logger.start_epoch()

        # Input is noise tensor, target is image
        with torch.set_grad_enabled(True):
            input_noise.to(args.device)
            logits = model.forward(input_noise)

            # Initialize loss to 0
            loss = torch.zeros(1, requires_grad=True).to(args.device)
            loss = loss_fn(logits, target_image.to(args.device))

            logger.log_status(input_noise, logits, target_image, loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = {'loss': loss.item()}
        saver.save(logger.epoch, model, optimizer, args.device, metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics)

if __name__ == "__main__":
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
