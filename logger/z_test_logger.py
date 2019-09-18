from time import time
from tqdm import tqdm

import util
from .base_logger import BaseLogger


class ZTestLogger(BaseLogger):
    """Class for logging z test info (training/backpropping z vector to fit test image) info to the console and saving model parameters to disk."""
    def __init__(self, args):
        super(ZTestLogger, self).__init__(args)

        self.experiment_name = args.name
        
        self.steps_per_print = args.steps_per_z_test_print
        self.steps_per_visual = args.steps_per_z_test_visual
        self.max_epochs = args.max_z_test_epochs
        self.convergence_loss = args.max_z_test_loss
        
        self.loss = 1.0
        self.loss_meter = util.AverageMeter()
        self.pbar = tqdm(total=self.max_epochs)
        self.train_start_time = time()

    def log_status(self, inputs, targets, probs, loss,
                   save_preds=False, force_visualize=False):
        """Log results and status of training."""
        
        batch_size = inputs.size(0)
        
        self.loss = loss.item()
        self.loss_meter.update(loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.global_step % self.steps_per_print == 0:

            # Write a header for the log entry
            duration = time() - self.train_start_time
            hours, rem = divmod(duration, 3600)
            minutes, seconds = divmod(rem, 60)
            
            message = f'[z-test][epoch: {self.epoch}, step: {self.global_step}, time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}, loss: {self.loss_meter.avg:.3g}]'
            self.pbar.set_description(message)
            
            # Write all errors as scalars to the graph
            self._log_scalars({'loss': self.loss_meter.avg}, print_to_stdout=False)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.global_step % self.steps_per_visual == 0 or force_visualize:
            self.visualize(probs, targets, probs, phase='z_test')

            if save_preds:
                probs_image_name = f'prediction-{epoch}.png'
                z_test_log_dir = os.path.join(self.log_dir, 'z_test')
                probs_image_path = os.path.join(z_test_log_dir, probs_image_name)
                
                probs = probs.detach().to('cpu')
                probs = probs.numpy().copy()
                probs_np = util.convert_image_from_tensor(probs)

                probs_pil = Image.fromarray(probs_np)
                probs_pil.save(probs_image_path)

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()

    def end_epoch(self):
        """Log info for end of an epoch.
        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            curves: Dictionary of curves. Items have format '{phase}_{curve}: value.
        """
        self.epoch += 1
        self.pbar.update(1)

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return (self.max_epochs < self.epoch) or (self.convergence_loss > self.loss)

