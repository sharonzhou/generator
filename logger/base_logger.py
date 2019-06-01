import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import util
from PIL import Image

from datetime import datetime
from tensorboardX import SummaryWriter

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
        log_dir = os.path.join('logs', args.name + '_' + datetime.now().strftime('%b%d_%H%M'))
        self.log_dir = log_dir

        self.summary_writer = SummaryWriter(log_dir=log_dir)

        # log_path, path to log, will be stored with the models
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(args.name))

        self.epoch = args.start_epoch


    def _log_text(self, text_dict):
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.epoch)


    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.epoch)

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
            self.summary_writer.add_image(name.replace('_', '/'), curve_img, global_step=self.epoch)

    def visualize(self, inputs, logits, targets, phase, epoch, unique_id=None, make_separate_prediction_img=False):
        """Visualize predictions and targets in TensorBoard.
        Args:
            inputs: Inputs to the model.
            logits: Logits predicted by the model.
            targets: Target labels for the inputs.
            phase: One of 'train', 'val', or 'test'.
            unique_id: A unique ID to append to every image title. Allows
              for displaying all visualizations separately on TensorBoard.
        Returns:
            Number of examples visualized to TensorBoard.
        """

        logits = logits.detach().to('cpu')
        logits = logits.numpy().copy()
        logits_np = util.convert_image_from_tensor(logits)

        targets = targets.detach().to('cpu')
        targets = targets.numpy().copy()
        targets_np = util.convert_image_from_tensor(targets)

        abs_diff = np.abs(targets_np - logits_np)

        visuals = [logits_np, targets_np, abs_diff]
        visuals_np = util.concat_images(visuals)
        visuals_pil = Image.fromarray(visuals_np)
        
        title = 'target_pred_diff'
        visuals_image_name = f'{title}-{epoch}.png'
        visuals_image_path = os.path.join(self.log_dir, visuals_image_name)
        
        visuals_pil.save(visuals_image_path)

        tag = f'{phase}/{title}'
        if unique_id is not None:
            tag += '_{}'.format(unique_id)

        self.summary_writer.add_image(tag, np.uint8(visuals_np), self.epoch)

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
