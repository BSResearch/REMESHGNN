from .base_options import BaseOptions


class ClassificationOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--checkpoints_dir', type=str, default='/media/bs/Seagate Backup Plus Drive/Project/Bahareh/Classification/checkpoints', help='models are saved here')
        # self.parser.add_argument('--dataset_mode', choices={"classification","CD_based_classification", "retrieval"}, default='LasConRetrieval')
        self.parser.add_argument('--loss_mode', choices={"weighted", "normal"}, default='normal')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=500,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.003, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='step',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=20,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        # use pretrained network
        self.parser.add_argument('--pretrained_model_file', type=str,
                                 default=None,
                                 help='pretrained model')
        self.is_train = True
