from time import time
from tqdm import tqdm

import util
from .base_logger import BaseLogger


class TestLogger(BaseLogger):
    """Class for logging testing info to the console and saving model parameters to disk."""
    def __init__(self, args):
        super(TestLogger, self).__init__(args)
        
        self.steps_per_print = args.steps_per_print
        self.steps_per_visual = args.steps_per_visual
        self.experiment_name = args.name
        self.num_epochs = args.num_epochs
        
        self.masked_loss_meter = util.AverageMeter()
        self.full_loss_meter = util.AverageMeter()
        self.obscured_loss_meter = util.AverageMeter()
        
        self.pbar = tqdm(total = int(self.num_epochs / self.steps_per_print))
        self.train_start_time = time()

    def log_hparams(self, args):
        """Log all the hyper parameters in tensorboard"""

        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({'hparams/' + key: args_dict[key]})

        self._log_text(hparams)

    def log_status(self, masked_probs, masked_loss, masked_test_target,
                   full_probs, full_loss, full_test_target,
                   obscured_probs, obscured_loss, obscured_test_target,
                   save_preds=False, force_visualize=False):
        """Log results and status of z test."""
        
        batch_size = full_probs.size(0)
        
        masked_loss = masked_loss.item()
        full_loss = full_loss.item()
        obscured_loss = obscured_loss.item()

        self.masked_loss_meter.update(masked_loss, batch_size)
        self.full_loss_meter.update(full_loss, batch_size)
        self.obscured_loss_meter.update(obscured_loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.global_step % self.steps_per_print == 0:

            # Write a header for the log entry
            duration = time() - self.train_start_time
            hours, rem = divmod(duration, 3600)
            minutes, seconds = divmod(rem, 60)
            
            message = f'[epoch: {self.epoch}, step: {self.global_step}, time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}, masked loss: {self.masked_loss_meter.avg:.3g}, full loss: {self.full_loss_meter.avg:.3g}, obscured_loss: {self.obscured_loss_meter.avg:.3g}]'
            self.pbar.set_description(message)
            self.pbar.update(1)

            # Write all errors as scalars to the graph
            self._log_scalars({'loss_masked': self.masked_loss_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_full': self.full_loss_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_obscured': self.obscured_loss_meter.avg}, print_to_stdout=False)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.global_step % self.steps_per_visual == 0 or force_visualize:
            
            self.visualize(full_probs, full_test_target, obscured_probs, phase='test')

            if save_preds:
                probs_image_name = f'z-test-pred-{self.global_step}.png'
                probs_image_path = os.path.join(self.log_dir, probs_image_name)
                
                full_probs = full_probs.detach().to('cpu')
                full_probs = full_probs.numpy().copy()
                full_probs_np = util.convert_image_from_tensor(full_probs)

                full_probs_pil = Image.fromarray(full_probs_np)
                full_probs_pil.save(probs_image_path)

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0

    def end_epoch(self):
        """Log info for end of an epoch.
        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            curves: Dictionary of curves. Items have format '{phase}_{curve}: value.
        """
        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch

    def start_iter(self):
        """Log info for start of an epoch."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an epoch.
        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            curves: Dictionary of curves. Items have format '{phase}_{curve}: value.
        """
        self.iter += self.batch_size
        self.global_step += self.batch_size
