import models
import os
import shutil
import torch
import torch.nn as nn


class ModelSaver(object):
    """Class to save and load model ckpts."""
    def __init__(self, args, maximize_metric=False):
        """
        Args:
            save_dir: Directory to save checkpoints.
            steps_per_save: Number of steps between each save.
            max_ckpts: Maximum number of checkpoints to keep before overwriting old ones.
            metric_name: Name of metric used to determine best model.
            maximize_metric: Bool of whether the objective is to maximize or minimize the metric.
            """
        super(ModelSaver, self).__init__()

        self.save_dir = args.save_dir
        self.steps_per_save = args.steps_per_save
        self.max_ckpts = args.max_ckpts
        self.metric_name = args.best_ckpt_metric
        self.maximize_metric = maximize_metric

        self.best_metric_val = None
        self.ckpt_paths = []

    def _is_best(self, metric_val):
        """Check whether metric_val is the best one we've seen so far."""
        if metric_val is None:
            return False
        return (self.best_metric_val is None
                or (self.maximize_metric and self.best_metric_val < metric_val)
                or (not self.maximize_metric and self.best_metric_val > metric_val))

    def save(self, global_step, model, optimizer, device, metric_val):
        """If this step corresponds to a save step, save model parameters to disk.
        Args:
            global_step: Iter to stamp on the checkpoint.
            model: Model to save.
            optimizer: Optimizer for model parameters.
            lr_scheduler: Learning rate scheduler for optimizer. (not included right now)
            device: Device where the model/optimizer parameters belong.
            metric_val: Value for determining whether checkpoint is best so far.
        """
        if global_step % self.steps_per_save != 0:
            return

        ckpt_dict = {
            'ckpt_info': {'global_step': global_step, self.metric_name: metric_val},
            'model_name': model.module.__class__.__name__,
            'model_args': model.module.args_dict(),
            'model_state': model.to('cpu').state_dict(),
            'optimizer': optimizer.state_dict(),
            #'lr_scheduler': lr_scheduler.state_dict()
        }
        model.to(device)

        ckpt_path = os.path.join(self.save_dir, 'step_{}.pth.tar'.format(global_step))
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_val):
            # Save the best model
            self.best_metric_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(ckpt_path, best_path)

        # Remove a checkpoint if more than max_ckpts ckpts saved
        self.ckpt_paths.append(ckpt_path)
        if len(self.ckpt_paths) > self.max_ckpts:
            oldest_ckpt = self.ckpt_paths.pop(0)
            os.remove(oldest_ckpt)

    @classmethod
    def load_model(ckpt_path, gpu_ids):
        """Load model parameters from disk.
        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for DataParallel.
        Returns:
            Model loaded from checkpoint, dict of additional checkpoint info (e.g. global_step, metric).
        """
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)

        # Build model, load parameters
        model_fn = models.__dict__[ckpt_dict['model_name']]
        model_args = ckpt_dict['model_args']
        model = model_fn(**model_args)
        model = nn.DataParallel(model, gpu_ids)
        model.load_state_dict(ckpt_dict['model_state'])

        return model, ckpt_dict['ckpt_info']

    @classmethod
    def load_optimizer(ckpt_path, optimizer, lr_scheduler=None):
        """Load optimizer and LR scheduler state from disk.
        Args:
            ckpt_path: Path to checkpoint to load.
            optimizer: Optimizer to initialize with parameters from the checkpoint.
            lr_scheduler: Optional learning rate scheduler to initialize with parameters from the checkpoint.
        """
        ckpt_dict = torch.load(ckpt_path)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
