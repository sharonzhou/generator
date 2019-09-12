import util

from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # Logging args
        self.parser.add_argument('--epochs_per_print', type=int, default=5,
                                 help='Number of epochs between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--epochs_per_eval', type=int, default=1,
                                 help='Number of epochs between evaluating model on the validation set.')
        self.parser.add_argument('--epochs_per_visual', type=int, default=50,
                                 help='Number of epochs between visualizing training examples.')
        self.parser.add_argument('--epochs_per_save', type=int, default=3,
                                 help='Number of epochs between saving a checkpoint to save_dir.')
        self.parser.add_argument('--max_ckpts', type=int, default=5,
                                 help='Number of recent ckpts to keep before overwriting old ones.')
        self.parser.add_argument('--best_ckpt_metric', type=str, default='masked_loss', choices=('masked_loss'),
                                 help='Metric used to determine which checkpoint is best.')

        # Learning rate args
        self.parser.add_argument('--learning_rate', type=float, default=3e-3,
                                 help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler', type=str, default='step', choices=('step', 'multi_step', 'plateau'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                                 help='Multiply learning rate by this value every LR step (step and multi_step only).')
        self.parser.add_argument('--lr_decay_step', type=int, default=100,
                                 help='Number of epochs between each multiply-by-gamma step.')
        self.parser.add_argument('--lr_milestones', type=str, default='50,125,250',
                                 help='Epochs to step the LR when using multi_step LR scheduler.')
        self.parser.add_argument('--patience', type=int, default=10,
                                 help='Number of stagnant epochs before stepping LR.')
        
        # Optimizer args
        self.parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum (SGD only).')
        self.parser.add_argument('--sgd_dampening', type=float, default=0.9, help='SGD momentum (SGD only).')
        self.parser.add_argument('--adam_beta_1', type=float, default=0.9, help='Adam beta 1 (Adam only).')
        self.parser.add_argument('--adam_beta_2', type=float, default=0.999, help='Adam beta 2 (Adam only).')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay (i.e., L2 regularization factor).')
        self.parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout probability.')
        self.parser.add_argument('--num_epochs', type=int, default=15000,
                                 help='Number of epochs to train. If 0, train forever.')

        # Loss args
        self.parser.add_argument('--loss_fn', type=str, default='mse', choices=('mse', 'cross_entropy'),
                                 help='Loss function to use.')
        
        # Model args
        self.parser.add_argument('--use_pretrained', action='store_true', help='If True, load a pretrained model from ckpt_path.')

        # Prediction args
        self.parser.add_argument('--save_preds', action='store_true', help='Save prediction every visualize step.')

        # Z-test args
        self.parser.add_argument('--epochs_per_z_test', type=int, default=500,
                                 help='Number of epochs between running z-test in the main training loop.')
        self.parser.add_argument('--max_z_test_epochs', type=int, default=1000,
                                 help='Stop criteria: max number of epochs to run z-test for during the z-test training loop.')
        self.parser.add_argument('--max_z_test_loss', type=float, default=0.00001,
                help='Convergence criteria: z loss at which we start saving masked/obscured values in the outer/main training loop.')
        self.parser.add_argument('--epochs_per_z_test_print', type=int, default=5,
                                 help='Number of epochs between running z-test print during the z-test training loop.')
        self.parser.add_argument('--epochs_per_z_test_visual', type=int, default=5,
                                 help='Number of epochs between displaying z-test visuals during the z-test training loop.')
