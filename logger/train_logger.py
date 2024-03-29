from time import time
from tqdm import tqdm

import util
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""
    def __init__(self, args):
        super(TrainLogger, self).__init__(args)
        """
        assert args.is_training
        assert args.steps_per_print % args.batch_size == 0, "steps_per_print must be divisible by batch_size"
        assert args.steps_per_visual % args.batch_size == 0, "steps_per_visual must be divisible by batch_size"
        """
        self.steps_per_print = args.steps_per_print
        self.steps_per_visual = args.steps_per_visual
        self.experiment_name = args.name
        self.num_epochs = args.num_epochs
        
        self.masked_loss_meter = util.AverageMeter()
        self.masked_loss_eval_meter = util.AverageMeter()
        self.obscured_loss_eval_meter = util.AverageMeter()
        self.full_loss_eval_meter = util.AverageMeter()
        self.z_loss_meter = util.AverageMeter()
        
        self.pbar = tqdm(total=args.num_epochs)
        self.train_start_time = time()

    def log_hparams(self, args):
        """Log all the hyper parameters in tensorboard"""

        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({'hparams/' + key: args_dict[key]})

        self._log_text(hparams)

    def log_status(self, inputs, targets, probs, masked_probs, masked_loss,
                   probs_eval, masked_probs_eval, obscured_probs_eval,
                   masked_loss_eval, obscured_loss_eval, full_loss_eval, 
                   z_target, z_probs, z_loss, save_preds=False, force_visualize=False):
        """Log results and status of training."""
        
        batch_size = inputs.size(0)
        
        masked_loss = masked_loss.item()
        masked_loss_eval = masked_loss_eval.item()
        obscured_loss_eval = obscured_loss_eval.item()
        full_loss_eval = full_loss_eval.item()
        z_loss = z_loss.item()

        self.masked_loss_meter.update(masked_loss, batch_size)
        self.masked_loss_eval_meter.update(masked_loss_eval, batch_size)
        self.obscured_loss_eval_meter.update(obscured_loss_eval, batch_size)
        self.full_loss_eval_meter.update(full_loss_eval, batch_size)
        self.z_loss_meter.update(z_loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.global_step % self.steps_per_print == 0:

            # Write a header for the log entry
            duration = time() - self.train_start_time
            hours, rem = divmod(duration, 3600)
            minutes, seconds = divmod(rem, 60)
            
            message = f'[epoch: {self.epoch}, step: {self.global_step}, time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}, masked loss (train): {self.masked_loss_meter.avg:.3g}, masked loss (eval): {self.masked_loss_eval_meter.avg:.3g}, obscured_loss: {self.obscured_loss_eval_meter.avg:.3g}, loss: {self.full_loss_eval_meter.avg:.3g}, z-loss: {self.z_loss_meter.avg:.3g}]'
            self.pbar.set_description(message)
            
            # Write all errors as scalars to the graph
            self._log_scalars({'loss_masked': self.masked_loss_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_masked-eval': self.masked_loss_eval_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_obscured': self.obscured_loss_eval_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_all': self.full_loss_eval_meter.avg}, print_to_stdout=False)
            self._log_scalars({'z_loss': self.z_loss_meter.avg}, print_to_stdout=False)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.global_step % self.steps_per_visual == 0 or force_visualize:
            # Does not make sense to show masked or obscured probs... since not image size anymore
            self.visualize(probs, targets, obscured_probs_eval, phase='train')
            self.visualize(probs_eval, targets, obscured_probs_eval, phase='eval')
            self.visualize(z_probs, z_target, z_probs, phase='z-test')

            if save_preds:
                probs_image_name = f'prediction-{self.global_step}.png'
                probs_image_path = os.path.join(self.log_dir, probs_image_name)
                
                probs = probs.detach().to('cpu')
                probs = probs.numpy().copy()
                probs_np = util.convert_image_from_tensor(probs)

                probs_pil = Image.fromarray(probs_np)
                probs_pil.save(probs_image_path)

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
        self.pbar.update(1)

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
