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
        assert args.epochs_per_print % args.batch_size == 0, "epochs_per_print must be divisible by batch_size"
        assert args.epochs_per_visual % args.batch_size == 0, "epochs_per_visual must be divisible by batch_size"
        """
        self.epochs_per_print = args.epochs_per_print
        self.epochs_per_visual = args.epochs_per_visual
        self.experiment_name = args.name
        self.num_epochs = args.num_epochs
        self.masked_loss_meter = util.AverageMeter()
        self.obscured_loss_meter = util.AverageMeter()
        self.loss_meter = util.AverageMeter()
        self.pbar = tqdm(total=args.num_epochs)
        self.train_start_time = time()

    def log_hparams(self, args):
        """Log all the hyper parameters in tensorboard"""

        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({'hparams/' + key: args_dict[key]})

        self._log_text(hparams)

    def log_status(self, inputs, masked_logits, obscured_logits, logits,
                   targets, masked_loss, obscured_loss, loss, 
                   save_preds=False, force_visualize=False):
        """Log results and status of training."""
        
        masked_loss = masked_loss.item()
        obscured_loss = obscured_loss.item()
        loss = loss.item()
        batch_size = inputs.size(0)

        self.masked_loss_meter.update(masked_loss, batch_size)
        self.obscured_loss_meter.update(obscured_loss, batch_size)
        self.loss_meter.update(loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.epoch % self.epochs_per_print == 0:

            # Write a header for the log entry
            duration = time() - self.train_start_time
            hours, rem = divmod(duration, 3600)
            minutes, seconds = divmod(rem, 60)
            
            message = f'[epoch: {self.epoch}, time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}, masked loss: {self.masked_loss_meter.avg:.3g}, obscured_loss: {self.obscured_loss_meter.avg:.3g}, loss: {self.loss_meter.avg:.3g}]'
            self.pbar.set_description(message)
            
            # Write all errors as scalars to the graph
            self._log_scalars({'loss_masked': self.masked_loss_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_obscured': self.obscured_loss_meter.avg}, print_to_stdout=False)
            self._log_scalars({'loss_all': self.loss_meter.avg}, print_to_stdout=False)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.epoch % self.epochs_per_visual == 0 or force_visualize:
            # Does not make sense to show masked or obscured logits... since not image size anymore
            self.visualize(inputs, logits, targets, phase='train', epoch=self.epoch)

            if save_preds:
                logits_image_name = f'prediction-{epoch}.png'
                logits_image_path = os.path.join(self.log_dir, logits_image_name)
                
                logits = logits.detach().to('cpu')
                logits = logits.numpy().copy()
                logits_np = util.convert_image_from_tensor(logits)

                logits_pil = Image.fromarray(logits_np)
                logits_pil.save(logits_image_path)

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
        return 0 < self.num_epochs < self.epoch

