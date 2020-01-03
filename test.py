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

from pytorch_pretrained_biggan import one_hot_from_int, truncated_noise_sample

def test(args):
    # Get loader for z-test
    loader = get_loader(args, phase='test')

    # TODO: make into function that takes in args.model and returns the pretrained model
    #       and also consider whether it's class conditional and what kind of class conditional (how many classes) -> probably just imagenet now, actually maybe cifar-10 too
    #       and also consider add truncation sampling as option too - this should return model, z_test noise vec, and class_vec (optionally)
    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    else:
        if 'BigGAN' in args.model:
            num_params = int(''.join(filter(str.isdigit, args.model)))
                
            if 'perturbation' in args.loss_fn:
                # Use custom BigGAN with Perturbation Net wrapper
                model = models.BigGANPerturbationNet.from_pretrained(f'biggan-deep-{num_params}')
            else:
                # Use pretrained BigGAN from package
                from pytorch_pretrained_biggan import BigGAN
                model = BigGAN.from_pretrained(f'biggan-deep-{num_params}')
    
    # Freeze model instead of using .eval()
    for param in model.parameters():
        param.requires_grad = False

    # If using perturbation net, learn perturbation layers
    if 'perturbation' in args.loss_fn:
        trainable_params = []
        for name, param in model.named_parameters():
            if 'perturb' in name:
                param.requires_grad = True
                trainable_params.append(param)
    
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    
    # Loss functions
    if 'mse' in args.loss_fn:
        pixel_criterion = torch.nn.MSELoss().to(args.device)
    else:
        pixel_criterion = torch.nn.L1Loss().to(args.device)

    if 'perceptual' in args.loss_fn:
        # Combination pixel-perceptual loss - Sec 3.2. By default, uses pixel L1.
        perceptual_criterion = torch.nn.L1Loss().to(args.device)
        perceptual_loss_weight = args.perceptual_loss_weight

        vgg_feature_extractor = models.VGGFeatureExtractor().to(args.device)
        vgg_feature_extractor.eval()
    elif 'perturbation' in args.loss_fn:
        # Perturbation network R. By default, uses pixel L1.
        # Sec 3.3: http://ganpaint.io/Bau_et_al_Semantic_Photo_Manipulation_preprint.pdf
        reg_criterion = torch.nn.MSELoss().to(args.device)
        reg_loss_weight = args.reg_loss_weight

    # z_loss_fn = util.get_loss_fn(args.loss_fn, args)
    max_z_test_loss = 100. # TODO: actually put max value possible here

    # Get logger, saver 
    logger = TestLogger(args)
    # saver = ModelSaver(args) TODO: saver for perturbation network R
    
    print(f'Logs: {logger.log_dir}')
    print(f'Ckpts: {args.save_dir}')

    # Run z-test in batches
    logger.log_hparams(args)
    batch_size = args.batch_size
    
    # Get noise vector
    # TODO: add truncation to args
    # truncation = args.truncation if args.truncation else 1.0
    truncation = 1.0
    z_test = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
    z_test = torch.from_numpy(z_test)
    
    # Get class conditional label
    # 981 is baseball player
    # 207 is golden retriever
    # TODO: Conditional generation only
    class_vector = one_hot_from_int(207, batch_size=batch_size)
    class_vector = torch.from_numpy(class_vector)

    while not logger.is_finished_training():
        logger.start_epoch()
         
        for _, z_test_target, mask in loader:
            logger.start_iter()
           
            if torch.cuda.is_available():
                mask = mask.cuda()
                z_test = z_test.cuda()
                z_test_target = z_test_target.cuda()
                class_vector = class_vector.cuda()
            
            masked_z_test_target = z_test_target * mask
            obscured_z_test_target = z_test_target * (1.0 - mask)
            
            if 'perturbation' in args.loss_fn:
                # With backprop on only trainable parameters in perturbation net
                z_optimizer = util.get_optimizer(trainable_params, args)
            else:
                # With backprop on only the input z, run one step of z-test and get z-loss
                z_optimizer = util.get_optimizer([z_test.requires_grad_()], args)

            with torch.set_grad_enabled(True):

                if class_vector is not None:
                    z_probs = model.forward(z_test, class_vector, truncation).float()
                    z_probs = (z_probs + 1) / 2.
                else:
                    z_probs = model.forward(z_test).float()
                    
                # Calculate the masked loss using z-test vector
                masked_z_probs = z_probs * mask
                z_loss = torch.zeros(1, requires_grad=True).to(args.device)

                pixel_loss = torch.zeros(1, requires_grad=True).to(args.device)
                pixel_loss = pixel_criterion(masked_z_probs, masked_z_test_target)

                if 'perceptual' in args.loss_fn:
                    z_probs_features = vgg_feature_extractor(masked_z_probs)
                    z_test_features = vgg_feature_extractor(masked_z_test_target).detach()
                    
                    perceptual_loss = torch.zeros(1, requires_grad=True).to(args.device)
                    perceptual_loss = perceptual_criterion(z_probs_features, z_test_features)

                    z_loss = pixel_loss + perceptual_loss_weight * perceptual_loss
                elif 'perturbation' in args.loss_fn:
                    # TODO: create perturbation network R - make it copy G fine layer shapes - need it to do forward pass during model.forward
                    # so first do model.forward of the usual model G on the high level layers, then replace with R? or interchange layers btw G and R and wrap the full model in a larger thing - where the G layers are frozen, but the R layers are not - I wonder if there's an easier way to do this so it's easy to retrofit any G - maybe if R copies all of G_finelayers, freezes itself on those params, then creates its own around each that's not frozen, that could work and the second forward pass is just R
                    reg_loss = reg_criterion() # TODO

                    z_loss = pixel_loss + reg_loss_weight * reg_loss
                else:
                    z_loss = pixel_loss
                
                # Backprop on z-test vector
                z_loss.backward()
                z_optimizer.step() 
                z_optimizer.zero_grad()

            # Compute the full loss (without mask) and obscured loss (loss only on masked region)
            # For logging and final evaluation (obscured loss is final MSE), so not in backprop loop
            full_z_loss = torch.zeros(1)
            full_pixel_loss = torch.zeros(1)
            full_pixel_loss = pixel_criterion(z_probs, z_test_target) #.mean()
            
            obscured_z_probs = z_probs * (1.0 - mask) 
            obscured_z_loss = torch.zeros(1)
            obscured_pixel_loss = torch.zeros(1)
            obscured_pixel_loss = pixel_criterion(obscured_z_probs, obscured_z_test_target) #.mean()
            
            if 'perceptual' in args.loss_fn: 
                # Full loss
                z_probs_full_features = vgg_feature_extractor(z_probs).detach()
                z_test_full_features = vgg_feature_extractor(z_test_target).detach()
                    
                full_perceptual_loss = torch.zeros(1)
                full_perceptual_loss = perceptual_criterion(z_probs_full_features, z_test_full_features)

                full_z_loss = full_pixel_loss + perceptual_loss_weight * full_perceptual_loss

                # Obscured loss
                z_probs_obscured_features = vgg_feature_extractor(z_probs).detach()
                z_test_obscured_features = vgg_feature_extractor(z_test_target).detach()
                    
                obscured_perceptual_loss = torch.zeros(1)
                obscured_perceptual_loss = perceptual_criterion(z_probs_obscured_features, z_test_obscured_features)

                obscured_z_loss = obscured_pixel_loss + perceptual_loss_weight * obscured_perceptual_loss
            elif 'pertubation' in args.loss_fn:
                reg_loss = reg_criterion() # TODO
                z_loss = pixel_loss + reg_loss_weight * reg_loss
            else:
                full_z_loss = full_pixel_loss
            
            """# TODO: z_loss is not always MSE anymore - figure out desired metric
            if z_loss < max_z_test_loss:
                # Save MSE on obscured region # TODO: z_loss is not always MSE anymore - figure out desired metric
                final_metrics = {'z-loss': z_loss.item(), 'obscured-z-loss': obscured_z_loss.item()}
                logger._log_scalars(final_metrics)
                print('Recall (z loss - non obscured loss - if MSE)', z_loss) 
                print('Precision (MSE value on masked region)', obscured_z_loss)
            """
            
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
