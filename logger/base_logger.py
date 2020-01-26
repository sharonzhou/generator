import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F

import util


plt.switch_backend('agg')


class BaseLogger(object):
    def __init__(self, args):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir

        self.num_visuals = args.num_visuals

        # log dir, is the directory for tensorboard logs: <project_dir>/logs/
        log_dir = os.path.join('logs', args.name + '_' + datetime.now().strftime('%b%d_%H%M%S%f'))
        self.log_dir = log_dir

        self.summary_writer = SummaryWriter(log_dir=log_dir)

        # log_path, path to log, will be stored with the models
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(args.name))

        self.epoch = args.start_epoch
        self.iter = args.start_iter
        self.global_step = args.start_iter


    def _log_text(self, text_dict):
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.global_step)


    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.global_step)

    def _plot_curves(self, curves_dict):
        """Plot all curves in a dict as RGB images to TensorBoard."""
        for name, curve in curves_dict.items():
            fig = plt.figure()
            ax = plt.gca()

            plot_type = name.split('_')[-1]
            ax.set_title(plot_type)
            if plot_type == 'PRC':
                precision, recall, _ = curve
                ax.step(recall, precision, color='b', alpha=0.2, where='post')
                ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
            elif plot_type == 'ROC':
                false_positive_rate, true_positive_rate, _ = curve
                ax.plot(false_positive_rate, true_positive_rate, color='b')
                ax.plot([0, 1], [0, 1], 'r--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
            else:
                ax.plot(curve[0], curve[1], color='b')

            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])

            fig.canvas.draw()

            curve_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            curve_img = curve_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.summary_writer.add_image(name.replace('_', '/'), curve_img, global_step=self.global_step)

    def debug_visualize(self, tensors, unique_suffix=None):
        """Visualize in TensorBoard.
        Args:
            tensors: Tensor or list of Tensors to be visualized.
        """
        # If only one tensor not in a list
        if not isinstance(tensors, list):
            tensors = [tensors]
        
        visuals = []
        for tensor in tensors:
            tensor = tensor.detach().to('cpu')
            tensor = tensor.numpy().copy()
            tensor_np = util.convert_image_from_tensor(tensor)
            visuals.append(tensor_np)

        if len(visuals) > 1:
            visuals_np = util.concat_images(visuals)
            visuals_pil = Image.fromarray(visuals_np)
        else:
            visuals_pil = Image.fromarray(visuals[0])

        title = 'debug'
        tag = f'{title}'
        if unique_suffix is not None:
            tag += '_{}'.format(unique_suffix)

        self.summary_writer.add_image(tag, np.uint8(visuals_np), self.global_step)

    def visualize(self, probs_batch, targets_batch, obscured_probs_batch, phase, unique_suffix=None, make_separate_prediction_img=False):
        """Visualize predictions and targets in TensorBoard.
        Args:
            probs_batch: Probabilities outputted by the model, in minibatch.
            targets_batch: Target labels for the inputs, in minibatch.
            phase: One of 'train', 'z-test' (during training), or 'test' (z-test eval alone).
            unique_suffix: A unique suffix to append to every image title. Allows
              for displaying all visualizations separately on TensorBoard.
        Returns:
            Number of examples visualized to TensorBoard.
        """

        probs_batch = probs_batch.detach().to('cpu')
        probs_batch = probs_batch.numpy().copy()

        targets_batch = targets_batch.detach().to('cpu')
        targets_batch = targets_batch.numpy().copy()
        
        obscured_probs_batch = obscured_probs_batch.detach().to('cpu')
        obscured_probs_batch = obscured_probs_batch.numpy().copy()
        
        batch_size = targets_batch.shape[0] # Do not use self.batch_size -- this is local, handling edge cases
        visual_indices = random.sample(range(batch_size), min(self.num_visuals, batch_size))
        for i in visual_indices:
            probs = probs_batch[i]
            targets = targets_batch[i]
            obscured_probs = obscured_probs_batch[i]
            
            probs_np = util.convert_image_from_tensor(probs)
            targets_np = util.convert_image_from_tensor(targets)
            obscured_probs_np = util.convert_image_from_tensor(obscured_probs)

            if phase == "z-test":
                visuals = [probs_np, targets_np]
                
                title = 'target_pred'
                visuals_image_name = f'{title}-{self.global_step}-{i}.png'
                log_dir_z_test = os.path.join(self.log_dir, 'z_test')
                os.makedirs(log_dir_z_test, exist_ok=True)
                visuals_image_path = os.path.join(log_dir_z_test, visuals_image_name)
            else:
                #abs_diff = np.abs(targets_np - probs_np)
                from PIL import ImageChops
                targets_pil = Image.fromarray(targets_np)
                probs_pil = Image.fromarray(probs_np)
                abs_diff = ImageChops.difference(targets_pil, probs_pil)
                abs_diff = np.array(abs_diff)

                visuals = [probs_np, targets_np, abs_diff, obscured_probs_np]
            
                title = 'pred_target_diff_obscured'
                visuals_image_name = f'{title}-{self.global_step}-{i}.png'
                log_dir_mask = os.path.join(self.log_dir, 'mask')
                os.makedirs(log_dir_mask, exist_ok=True)
                visuals_image_path = os.path.join(log_dir_mask, visuals_image_name)
                
            visuals_np = util.concat_images(visuals)
            visuals_pil = Image.fromarray(visuals_np)
           
            if make_separate_prediction_img:
                visuals_pil.save(visuals_image_path)
            
            tag = f'{phase}/{title}'
            if unique_suffix is not None:
                tag += '_{}'.format(unique_suffix)
          
            # If channel dimension is not first, then move to front
            if visuals_np.shape[0] != 3 and visuals_np.shape[2] == 3:
                visuals_np = np.transpose(visuals_np, (2, 0, 1))
            self.summary_writer.add_image(tag, np.uint8(visuals_np), self.global_step)

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError
