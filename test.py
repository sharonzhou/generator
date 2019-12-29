"""
test.py

Run z-test on a pretrained model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

import util
import models
from data import get_loader
from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver

def test(args):
    # Get loader for z-test
    loader = get_loader(args, phase='test')

    # Load pretrained model ckpt by path or torchvision pretrained
    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    else:
        from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample,
                                                       save_as_images, display_in_terminal)
        model = BigGAN.from_pretrained('biggan-deep-512')
        """
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path)
        """
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.eval()

    # Print model parameters
    print('Model parameters: name, size, mean, std')
    for name, param in model.named_parameters():
        print(name, param.size(), torch.mean(param), torch.std(param))

    # Get optimizer and loss
    parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
  
    z_loss_fn = util.get_loss_fn(args.loss_fn, args)

    # Get logger, saver 
    logger = TestLogger(args)
    saver = ModelSaver(args)
    
    print(f'Logs: {logger.log_dir}')
    print(f'Ckpts: {args.save_dir}')

    # Run z-test in batches
    logger.log_hparams(args)
    batch_size = args.batch_size
    while not logger.is_finished_training():
        logger.start_epoch()
       
        for z_test, z_test_target, mask in loader:
            z_test = truncated_noise_sample(truncation=1.0, batch_size=1)
            z_test = torch.from_numpy(z_test)
            
            class_vector = one_hot_from_int(1, batch_size=1)
            class_vector = torch.from_numpy(class_vector)

            logger.start_iter()
           
            if torch.cuda.is_available():
                mask = mask.cuda()
                z_test = z_test.cuda()
                z_test_target = z_test_target.cuda()
                class_vector = class_vector.cuda()
            
            masked_z_test_target = z_test_target * mask
            obscured_z_test_target = z_test_target * (1.0 - mask)
            
            # With backprop on only the input z, run one step of z-test and get z-loss
            z_optimizer = util.get_optimizer([z_test.requires_grad_()], args)
            with torch.set_grad_enabled(True):
                if args.use_intermediate_logits:
                    z_logits = model.forward(z_test).float()
                    z_probs = F.sigmoid(z_logits)
                    
                    # Debug logits and diffs
                    logger.debug_visualize([z_logits, z_logits * mask, z_logits * (1.0 - mask)],
                                           unique_suffix='z-logits')
                else:
                    #z_probs = model.forward(z_test).float()
                    z_probs = model.forward(z_test, class_vector, 1.0).float()

                # Calculate the masked loss using z-test vector
                masked_z_probs = z_probs * mask
                z_loss = torch.zeros(1, requires_grad=True).to(args.device)
                z_loss = z_loss_fn(masked_z_probs, masked_z_test_target).mean()
                
                # Backprop on z-test vector
                z_loss.backward()
                z_optimizer.step() 
                z_optimizer.zero_grad()

            # Compute the full loss (without mask) and obscured loss (loss only on masked region)
            # For logging and final evaluation (obscured loss is final MSE), so not in backprop loop
            full_z_loss = torch.zeros(1)
            full_z_loss = z_loss_fn(z_probs, z_test_target).mean()
            
            obscured_z_probs = z_probs * (1.0 - mask) 
            obscured_z_loss = torch.zeros(1)
            obscured_z_loss = z_loss_fn(obscured_z_probs, obscured_z_test_target).mean()
            
            # Once the unmasked region starts to overfit, get score from obscured region
            if z_loss < args.max_z_test_loss: # TODO later: include this part into the metrics/saver stuff below
                # Save MSE on obscured region
                final_metrics = {'z-loss': z_loss.item(), 'obscured-z-loss': obscured_z_loss.item()}
                logger._log_scalars(final_metrics)
                print('z loss', z_loss) 
                print('Final MSE value', obscured_z_loss)

            # TODO later: Make a function for metrics - or at least make sure dict includes all possible best ckpt metrics
            # TODO: figure out criteria for saving - and technically just saving z-test value for each image? Model doesn't 
            # change don't need to save the model - same ckpt
            metrics = {'masked_loss': z_loss.item()}
            #saver.save(logger.global_step, model, optimizer, args.device, metric_val=metrics.get(args.best_ckpt_metric, None))        
            
            # Log both train and eval model settings, and visualize their outputs
            logger.log_status(masked_probs=masked_z_probs,
                              masked_loss=z_loss,
                              masked_test_target=masked_z_test_target,
                              full_probs=z_probs,
                              full_loss=full_z_loss,
                              full_test_target=z_test_target,
                              obscured_probs=obscured_z_probs,
                              obscured_loss=obscured_z_loss,
                              obscured_test_target=obscured_z_test_target,
                              save_preds=args.save_preds,
                              ) 

            logger.end_iter()
        
        logger.end_epoch()
    
    # Last log after everything completes
    logger.log_status(masked_probs=masked_z_probs,
                      masked_loss=z_loss,
                      masked_test_target=masked_z_test_target,
                      full_probs=z_probs,
                      full_loss=full_z_loss,
                      full_test_target=z_test_target,
                      obscured_probs=obscured_z_probs,
                      obscured_loss=obscured_z_loss,
                      obscured_test_target=obscured_z_test_target,
                      save_preds=args.save_preds,
                      force_visualize=True,
                      ) 


if __name__ == "__main__":
    parser = TestArgParser()
    args_ = parser.parse_args()
    test(args_)
