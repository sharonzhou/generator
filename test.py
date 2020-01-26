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
import os
from PIL import Image
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
from datetime import datetime
import pandas as pd

import util
import models
from data import get_loader
from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_inverted_net(args):
    # Start by training an external model on samples of G(z) -> z inversion
    model = util.get_invert_model(args)
    
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    print(f'{args.invert_model} num params {count_parameters(model)}')
    
    generator = util.get_model(args)
    if generator is not None:
        generator = nn.DataParallel(generator, args.gpu_ids)
        generator = generator.to(args.device)
        print(f'{args.model} num params {count_parameters(generator)}')
    else:
        # Load saved pairings (ProGAN/StyleGAN)
        pairing_dir = '/deep/group/gen-eval/model-training/src/GAN_models/stylegan'
        pairing_path = f'{pairing_dir}/otavio_sampled_output/pairing.csv'
        pairings = pd.read_csv(pairing_path)

        num_pairings = len(pairings)
        noise_targets = pairings['noise']
        image_inputs = pairings['image']

    if 'BigGAN' in args.model:
        class_vector = one_hot_from_int(207, batch_size=args.batch_size)
        class_vector = torch.from_numpy(class_vector)
        class_vector = class_vector.cuda()

    # TODO: remove bc cant use gpu in laoder i don't think
    #loader = get_loader(args, phase='invert')
    
    #logger = TestLogger(args)
    #logger.log_hparams(args)
    
    criterion = torch.nn.MSELoss().to(args.device)
    optimizer = util.get_optimizer(model.parameters(), args)

    for i in range(args.num_invert_epochs):
        if generator is not None:
            noise_target = util.get_noise(args) 
            
            image_input = generator.forward(noise_target).float()
            image_input = (image_input + 1.) / 2.
        else:
            # TODO: make into loader
            idx = i % num_pairings
            noise_target = np.load(f'{pairing_dir}/{noise_targets[idx]}')
            noise_target = torch.from_numpy(noise_target).float()
            print(f'noise target shape {noise_target.shape}')

            image_input = np.array(Image.open(f'{pairing_dir}/{image_inputs[idx]}'))
            image_input = torch.from_numpy(image_input / 255.)
            image_input = image_input.float().unsqueeze(0)
            image_input = image_input.permute(0, 3, 1, 2)
        
        noise_target = noise_target.cuda()
        image_input = image_input.cuda()

        with torch.set_grad_enabled(True):
            probs = model.forward(image_input)
                   
            loss = torch.zeros(1, requires_grad=True).to(args.device)
            loss = criterion(probs, noise_target)
            print(f'iter {i}: loss = {loss}')

            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

        if i % 1 == 0:
            corres_image_input = image_input.detach().cpu()
            corres_np = util.convert_image_from_tensor(corres_image_input)

            # Run check - saving image
            if 'BigGAN' in args.model:
                predicted_image = generator.forward(probs, class_vector, truncation).float()
            else:
                if generator is not None:
                    predicted_image = generator.forward(probs).float()
                    
                    predicted_image = predicted_image.detach().cpu()
                    predicted_image = (predicted_image + 1) / 2.
                    predicted_np = util.convert_image_from_tensor(predicted_image)

                    if len(predicted_np.shape) == 4:
                        predicted_np = predicted_np[0]
                        corres_np = corres_np[0]
                    visuals = util.concat_images([predicted_np, corres_np])
                    visuals_pil = Image.fromarray(visuals)
                    timestamp = datetime.now().strftime('%b%d_%H%M%S%f')
                    visuals_image_dir = f'predicted_inversion_images/{args.model}'
                    os.makedirs(visuals_image_dir, exist_ok=True)
                    visuals_image_path = f'{visuals_image_dir}/{timestamp}_{i}.png'
                    visuals_pil.save(visuals_image_path)

                    print(f'Saved {visuals_image_path}')
                else:
                    # Save noise vector - do forward separately in tf env
                    probs = probs.detach().cpu().numpy()
                    pred_noise_dir = f'predicted_inversion_noise/{args.model}'
                    os.makedirs(pred_noise_dir, exist_ok=True)
                    
                    pred_noise_path = f'{pred_noise_dir}/{args.model}_noise_{i}.npy'
                    np.save(pred_noise_path, probs)

                    print(f'Saved {pred_noise_path}')

        if i % 1 == 0:
            corres_image_input = image_input.detach().cpu()
            corres_np = util.convert_image_from_tensor(corres_image_input)
            
            if len(corres_np.shape) == 4:
                corres_np = corres_np[0]

            corres_pil = Image.fromarray(corres_np)
            timestamp = datetime.now().strftime('%b%d_%H%M%S%f')
            corres_image_dir = f'generated_images/{args.model}'
            os.makedirs(corres_image_dir, exist_ok=True)
            corres_image_path = f'{corres_image_dir}/{timestamp}_{i}.png'
            corres_pil.save(corres_image_path)
        
    # saver = ModelSaver(args)
    global_step = args.num_invert_epochs
    ckpt_dict = {
        'ckpt_info': {'global_step': global_step},
        'model_name': model.module.__class__.__name__,
        'model_args': model.module.args_dict(),
        'model_state': model.to('cpu').state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    ckpt_dir = os.path.join(args.save_dir, f'{args.model}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.invert_model}_step_{global_step}.pth.tar')
    torch.save(ckpt_dict, ckpt_path)
    print(f'Saved model to {ckpt_path}')

    import pdb;pdb.set_trace()

    return model 

def evaluate_inversion(args, inverted_net_path):
    # Load saved inverted net
    device = 'cuda:{}'.format(args.gpu_ids[0]) if len(args.gpu_ids) > 0 else 'cpu'
    ckpt_dict = torch.load(inverted_net_path, map_location=device)

    # Build model, load parameters
    model_args = ckpt_dict['model_args']
    inverted_net = models.ResNet18(**model_args)
    inverted_net = nn.DataParallel(inverted_net, args.gpu_ids)
    inverted_net.load_state_dict(ckpt_dict['model_state'])

    import pdb;pdb.set_trace()

    # Get test images (CelebA)
    initial_generated_image_dir = '/deep/group/sharonz/generator/z_test_images/'
    initial_generated_image_name = '058004_crop.jpg'
    initial_generated_image = util.get_image(initial_generated_image_dir, initial_generated_image_name)
    initial_generated_image = initial_generated_image / 255.
    intiial_generated_image = initial_generated_image.cuda()

    inverted_noise = inverted_net(initial_generated_image)
    
    if 'BigGAN' in args.model:
        class_vector = one_hot_from_int(207, batch_size=batch_size)
        class_vector = torch.from_numpy(class_vector)
    
        num_params = int(''.join(filter(str.isdigit, args.model)))
        generator = BigGAN.from_pretrained(f'biggan-deep-{num_params}')
    
        generator = generator.to(args.device)
        generated_image = generator.forward(inverted_noise, class_vector, args.truncation)
    
    # Get difference btw initial and subsequent generated image
    # Save both

    return

def test(args):
    # Get loader for z-test
    loader = get_loader(args, phase='test')
    batch_size = args.batch_size
    class_vector = None

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
                model = BigGAN.from_pretrained(f'biggan-deep-{num_params}')
            
            z_test = truncated_noise_sample(truncation=args.truncation, batch_size=batch_size)
            z_test = torch.from_numpy(z_test)
            
            # Get class conditional label
            # 981 is baseball player
            # 207 is golden retriever
            # TODO: Conditional generation only
            class_vector = one_hot_from_int(207, batch_size=batch_size)
            class_vector = torch.from_numpy(class_vector)
       
        elif 'WGAN-GP' in args.model:
            generator_path =  "/deep/group/gen-eval/model-training/src/GAN_models/improved-wgan-pytorch/experiments/exp4_wgan_gp/generator.pt"
            model = torch.load(generator_path)
            z_test = torch.randn(batch_size, 128)

        elif 'BEGAN' in args.model:
            generator_path = "/deep/group/gen-eval/model-training/src/GAN_models/BEGAN-pytorch/trained_models/64/models/gen_97000.pth"
            model = models.BEGANGenerator() 
            model.load_state_dict(torch.load(generator_path))

            z_test = np.random.uniform(-1, 1, size=(batch_size, 64))
            z_test = torch.FloatTensor(z_test)
    
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
        print(f'Number of trainable params: {len(trainable_params)}')
    
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
    
    while not logger.is_finished_training():
        logger.start_epoch()
         
        for _, z_test_target, mask in loader:
            logger.start_iter()
            if torch.cuda.is_available():
                mask = mask.cuda()
                z_test = z_test.cuda()
                z_test_target = z_test_target.cuda()
                #class_vector = class_vector.cuda()
            
            masked_z_test_target = z_test_target * mask
            obscured_z_test_target = z_test_target * (1.0 - mask)
            
            if 'perturbation' in args.loss_fn:
                # With backprop on only trainable parameters in perturbation net
                params = trainable_params + [z_test.requires_grad_()]
                z_optimizer = util.get_optimizer(params, args)
            else:
                # With backprop on only the input z, run one step of z-test and get z-loss
                z_optimizer = util.get_optimizer([z_test.requires_grad_()], args)

            with torch.set_grad_enabled(True):

                if class_vector is not None:
                    z_probs = model.forward(z_test, class_vector, args.truncation).float()
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
                    reg_loss = torch.zeros(1, requires_grad=True).to(args.device)
                    for name, param in model.named_parameters():
                        if 'perturb' in name:
                            delta = param - 1
                            reg_loss += torch.pow(delta, 2).mean() #sum()
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
            elif 'perturbation' in args.loss_fn:
                full_z_loss = full_pixel_loss + reg_loss_weight * reg_loss
                obscured_z_loss = obscured_pixel_loss + reg_loss_weight * reg_loss
            else:
                full_z_loss = full_pixel_loss
                obscured_z_loss = obscured_pixel_loss
            
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
    #test(args_)

    train_inverted_net(args_)
    #inverted_net = train_inverted_net(args_)
    #test(args_, inverted_net)
