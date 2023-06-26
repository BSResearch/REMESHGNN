from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='val, test')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--model_file', help='path to saved model')
        self.parser.add_argument('--pretrained_model_file', type=str, default=None,
                                 help='pretrained model')

        self.is_train = False
