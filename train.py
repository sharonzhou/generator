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

import util
import models
from args import TrainArgParser
from logger import TrainLogger
from saver import ModelSaver

def train(args):
    
    # Instantiate input and target: Input is noise tensor, target is image
    target_image = util.get_target_image(args).float()
    input_noise = util.get_input_noise(args)
    input_noise.requires_grad = True

    # Instantiate mask
    setattr(args, 'target_image_shape', target_image.shape)
    mask = util.get_mask(args)
    setattr(args, 'mask', mask)
    
    # For deep decoder net input, reshape input noise and do not parallelize
    if args.model == 'DeepDecoderNet':
        if args.use_custom_input_noise:
            input_noise = util.get_deep_decoder_input_noise(target_image.shape)
        else:
            # Do not parallelize (reshape noise has issues in fc layer), use 1st gpu
            gpu_ids = args.gpu_ids
            gpu_id = [gpu_ids[0]]
            setattr(args, 'gpu_ids', gpu_id)

    print(f'Input: {input_noise.shape}')
    print(f'Target: {target_image.shape}')
    if mask is not None:
        print(f'Mask: {mask.shape}')
        masked_target_image = target_image * mask
        obscured_target_image = target_image * (1.0 - mask)

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
        input_noise.detach().to('cpu').float()
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
        
        # TODO: Make a function for metrics - or at least make sure dict includes all possible best ckpt metrics
        metrics = {'masked_loss': masked_loss.item()}
        saver.save(logger.epoch, model, optimizer, args.device, metric_val=metrics.get(args.best_ckpt_metric, None))         
        
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
        model.train()

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
                          save_preds=args.save_preds,
                          )

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
                      save_preds=args.save_preds,
                      force_visualize=True,
                      )


if __name__ == "__main__":
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
