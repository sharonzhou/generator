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
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

import util
import models
from data import get_loader
from args import TrainArgParser
from logger import TrainLogger
from saver import ModelSaver

def train(args):
    # Get loader for outer loop training
    loader = get_loader(args)
    target_image_shape = loader.dataset.target_image_shape
    setattr(args, 'target_image_shape', target_image_shape)

    # Load model
    model_fn = models.__dict__[args.model]
    model = model_fn(**vars(args))
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Print model parameters
    print('Model parameters: name, size, mean, std')
    for name, param in model.named_parameters():
        print(name, param.size(), torch.mean(param), torch.std(param))

    # Get optimizer and loss
    parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    loss_fn = util.get_loss_fn(args.loss_fn, args)
  
    z_loss_fn = util.get_loss_fn(args.loss_fn, args)

    # Get logger, saver 
    logger = TrainLogger(args)
    saver = ModelSaver(args)
    
    print(f'Logs: {logger.log_dir}')
    print(f'Ckpts: {args.save_dir}')

    # Train model
    logger.log_hparams(args)
    batch_size = args.batch_size
    while not logger.is_finished_training():
        logger.start_epoch()
        
        for input_noise, target_image, mask, z_test_target, z_test in loader: 
            logger.start_iter()
           
            if torch.cuda.is_available():
                input_noise = input_noise.to(args.device) #.cuda()
                target_image = target_image.cuda()
                mask = mask.cuda()
                z_test = z_test.cuda()
                z_test_target = z_test_target.cuda()
            
            masked_target_image = target_image * mask
            obscured_target_image = target_image * (1.0 - mask)

            # Input is noise tensor, target is image
            model.train()
            with torch.set_grad_enabled(True):
                if args.use_intermediate_logits:
                    logits = model.forward(input_noise).float()
                    probs = F.sigmoid(logits) 

                    # Debug logits and diffs
                    logger.debug_visualize([logits, logits * mask, logits * (1.0 - mask)],
                                           unique_suffix='logits-train')
                else:
                    probs = model.forward(input_noise).float()

                # With backprop, calculate (1) masked loss, loss when mask is applied.
                # Loss is done elementwise without reduction, so must take mean after.
                # Easier for debugging.
                masked_probs = probs * mask
                masked_loss = torch.zeros(1, requires_grad=True).to(args.device)
                masked_loss = loss_fn(masked_probs, masked_target_image).mean()

                masked_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Without backprop, calculate (2) full loss on the entire image,
            # And (3) the obscured loss, region obscured by mask.
            model.eval()
            with torch.no_grad():
                if args.use_intermediate_logits:
                    logits_eval = model.forward(input_noise).float()
                    probs_eval = F.sigmoid(logits_eval) 

                    # Debug logits and diffs
                    logger.debug_visualize([logits_eval, logits_eval * mask, logits_eval * (1.0 - mask)],
                                           unique_suffix='logits-eval')
                else:
                    probs_eval = model.forward(input_noise).float()

                masked_probs_eval = probs_eval * mask
                masked_loss_eval = torch.zeros(1)
                masked_loss_eval = loss_fn(masked_probs_eval, masked_target_image).mean()

                full_loss_eval = torch.zeros(1)
                full_loss_eval = loss_fn(probs_eval, target_image).mean()
                
                obscured_probs_eval = probs_eval * (1.0 - mask) 
                obscured_loss_eval = torch.zeros(1)
                obscured_loss_eval = loss_fn(obscured_probs_eval, obscured_target_image).mean()
            
            # With backprop on only the input z, (4) run one step of z-test and get z-loss
            z_optimizer = util.get_optimizer([z_test.requires_grad_()], args)
            with torch.set_grad_enabled(True):
                if args.use_intermediate_logits:
                    z_logits = model.forward(z_test).float()
                    z_probs = F.sigmoid(z_logits) 
                else:
                    z_probs = model.forward(z_test).float()

                z_loss = torch.zeros(1, requires_grad=True).to(args.device)
                z_loss = z_loss_fn(z_probs, z_test_target).mean()

                z_loss.backward()
                z_optimizer.step() 
                z_optimizer.zero_grad()
                
            if z_loss < args.max_z_test_loss: # TODO: include this part into the metrics/saver stuff below
                # Save MSE on obscured region
                final_metrics = {'final/score': obscured_loss_eval.item()}
                logger._log_scalars(final_metrics)
                print('z loss', z_loss) 
                print('Final MSE value', obscured_loss_eval) 

            # TODO: Make a function for metrics - or at least make sure dict includes all possible best ckpt metrics
            metrics = {'masked_loss': masked_loss.item()}
            saver.save(logger.global_step, model, optimizer, args.device, metric_val=metrics.get(args.best_ckpt_metric, None))        
            # Log both train and eval model settings, and visualize their outputs
            logger.log_status(inputs=input_noise,
                              targets=target_image,
                              probs=probs,
                              masked_probs=masked_probs,
                              masked_loss=masked_loss,
                              probs_eval=probs_eval,
                              masked_probs_eval=masked_probs_eval,
                              obscured_probs_eval=obscured_probs_eval,
                              masked_loss_eval=masked_loss_eval,
                              obscured_loss_eval=obscured_loss_eval,
                              full_loss_eval=full_loss_eval,
                              z_target=z_test_target,
                              z_probs=z_probs,
                              z_loss=z_loss,
                              save_preds=args.save_preds,
                              ) 

            logger.end_iter() 
        
        logger.end_epoch()
    
    # Last log after everything completes
    logger.log_status(inputs=input_noise,
                      targets=target_image,
                      probs=probs,
                      masked_probs=masked_probs,
                      masked_loss=masked_loss,
                      probs_eval=probs_eval,
                      masked_probs_eval=masked_probs_eval,
                      obscured_probs_eval=obscured_probs_eval,
                      masked_loss_eval=masked_loss_eval,
                      obscured_loss_eval=obscured_loss_eval,
                      full_loss_eval=full_loss_eval,
                      z_target=z_test_target,
                      z_probs=z_probs,
                      z_loss=z_loss,
                      save_preds=args.save_preds,
                      force_visualize=True,
                      )


if __name__ == "__main__":
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
