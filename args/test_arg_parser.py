import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--phase', type=str, default='valid', 
                                 choices=('train', 'valid', 'test'),
                                 help='Phase to test on.')

        self.parser.add_argument('--results_dir', type=str, default='results/', help='Save dir for test results.')
        self.parser.add_argument('--save_visuals', action='store_true', default=False, help='If true, saves visualizations of the segmentation output for each patch in png file.')
