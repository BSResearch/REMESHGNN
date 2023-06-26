import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot',
                                 default ='/home/bs/Datasets/Retrieval/ShapeNetManifold_simplified/ShapeNetManifold_10000_simplified',
                                 help='path to graphs (should have sub-folders train, test)')
        # self.parser.add_argument('--ninput_edges', type=int, default=2250,
        #                          help='# of input edges (will include dummy edges)')
        # self.parser.add_argument('--dataset_mode', choices={"classification", "retrieval"}, default='classification')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples per epoch')
        self.parser.add_argument('--use_model', default ='nn.clf_layer_concat_30', help = 'model to use')

        # network params
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--embedding_hidden_size', type=int, default=[0, 0],
                                 help='input hidden size for'
                                      'initial encoding layer fully connected layer.'
                                      'Note: The list consist of the hidden layers sizes from first to the second last'
                                      'The last layer has the size which is equal to the fist layer of '
                                      'HyNet_hidden_size')

        self.parser.add_argument('--MeshConv',
                                default=[(10, 5, 64, 64, 64, 64, True, False),
                                          (70, 64, 64, 64, 64, 64, True, False),
                                          (70, 64, 64, 64, 64, 64, True, False),
                                          (70, 64, 64, 64, 64, 64, True, False),
                                          (70, 64, 64, 64, 64, 64, True, False),
                                          (70, 64, 64, 64, 64, 64, True, False)
                                          # (70, 64, 128, 64, 64, 128, True, False),
                                          # (70, 64, 128, 64, 64, 128, True, False)
                                          ],
                                 help='hidden size for series of MeshConv layers')
        self.parser.add_argument('--graph_classifier', type=int, default=[128,128],
                                 help='input hidden size for scoring edges to pool')

        self.parser.add_argument('--graph_classifier_dropout', type=float, default=0.5, help='GAT layer drop out')
        self.parser.add_argument('--gnn_dropout', type=float, default=0.5, help='GNN layer drop out')
        # general params
        self.parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='retrieval_chair_shapenet',
                                 help='name of the experiment.')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        self.parser.add_argument('--sbin', type=int,  default=8, help='if specified, uses seed')
        self.parser.add_argument('--save_features', type=bool, default=False,
                                 help='True if segmentation/classification result '
                                      'is required')
        self.parser.add_argument('--dataset_name', type=str, default='ModelNet40', choices={'ModelNet40','SHREC_Split16',
                                                                                            'SHREC_Split10'})
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        # if len(self.opt.gpu_ids) > 0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
