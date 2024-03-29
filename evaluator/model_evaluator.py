"""
Evaluator class
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy

import util
from logger import ZTestLogger

class Evaluator(object):
    """Evaluator class to perform two evaluations:
        (1) Mask: Evaluate masked, obscured, and full losses
        (2) Z-test: Backprop Z vector to held out test image(s)
    """
    def __init__(self, args, input_noise, target_image, mask, logger, **kwargs):
        # Parameters for both
        self.input_noise = input_noise
        self.logger = logger

        # Z-test only
        # Get the same loss fn
        # TODO: play with different loss fns for this component
        self.z_test_loss_fn = util.get_loss_fn(args.loss_fn, args)
        
        # Mask evaluation
        self.mask = mask
        self.target_image = target_image
        self.masked_target_image = self.target_image * self.mask
        self.obscured_target_image = self.target_image * (1.0 - self.mask)


    def evaluate_z_test(self, args, z_test, target, model, **kwargs):
        og_z_test = deepcopy(z_test)
        print('does requires grad come iwth?', z_test)


        loss = torch.ones(1, requires_grad=True).to(args.device)
        if self.logger.global_step % args.steps_per_z_test == 0: 
            # Will need to backprop into the input
            z_test = z_test.requires_grad_()

            # Optimizer with trainable input parameters
            optimizer = util.get_optimizer([z_test], args)
            
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Use model in eval mode for dropout/BN layers to behave their best
            model.eval()

            with torch.set_grad_enabled(True):
                if args.use_intermediate_logits:
                    logits = model.forward(z_test).float()
                    probs = F.sigmoid(logits) 
                else:
                    probs = model.forward(z_test).float()

                loss = self.z_test_loss_fn(probs, target).mean()

                loss.backward()
                optimizer.step() 
                optimizer.zero_grad()
                
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True

            model.train()
        
        print('sanity ztest', torch.sum(og_z_test), torch.sum(z_test))
        
        return z_test, loss #TODO! debug that z-test actually changes...

    """
    def evaluate_mask(self, batch_size, model, probs):

        model.eval()
        with torch.no_grad():
            if self.use_intermediate_logits:
                logits_eval = model.forward(self.input_noise).float()
                probs_eval = F.sigmoid(logits_eval) 

                # Debug logits and diffs
                self.logger.debug_visualize([logits_eval, logits_eval * self.mask, logits_eval * (1.0 - self.mask)],
                                       unique_suffix='logits-eval')
            else:
                probs_eval = model.forward(self.input_noise).float()

            masked_probs_eval = probs_eval * self.mask
            masked_loss_eval = torch.zeros(1)
            masked_loss_eval = self.loss_fn(masked_probs_eval, self.masked_target_image).mean()
            # self.logger.update_loss_meter(batch_size, 

            full_loss_eval = torch.zeros(1)
            full_loss_eval = self.loss_fn(probs_eval, self.target_image).mean()
            
            obscured_probs_eval = probs_eval * (1.0 - self.mask) 
            obscured_loss_eval = torch.zeros(1)
            obscured_loss_eval = self.loss_fn(obscured_probs_eval, self.obscured_target_image).mean()
        model.train()
        
        # Log both train and eval model settings, and visualize their outputs
        self.logger.log_status(inputs=self.input_noise,
                               targets=self.target_image,
                               probs=probs,
                               masked_loss=masked_loss,
                               probs_eval=probs_eval,
                               obscured_probs_eval=obscured_probs_eval,
                               masked_loss_eval=masked_loss_eval,
                               obscured_loss_eval=obscured_loss_eval,
                               full_loss_eval=full_loss_eval,
                               save_preds=self.save_preds,
                               )

        return model
    """
