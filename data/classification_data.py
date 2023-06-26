import torch
from .base_dataset import BaseDataset
import os
from util.util import is_graph_file
import numpy as np
import dgl
import pandas as pd
import pymeshlab


class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        # self.dataset_info_csv = os.path.join(opt.dataroot, 'all.csv')
        self.classes, self.class_to_idx = self.find_classes(self.root)
        if self.opt.dataset_name == 'ModelNet40' or self.opt.dataset_name == 'SHREC_Split16':
            self.paths, self.labels = self.make_dataset_by_class(self.root,  self.class_to_idx, opt.phase)
        if self.opt.dataset_name == 'SHREC_Split10':
            self.paths, self.labels = self.make_SHREC_S10_dataset_by_class(self.root, self.class_to_idx)

        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # self.canonical_etypes = self.get_canoncial_etypes()
        # self.offset = self.classes[0]
        opt.nclasses = self.nclasses
        # self.paths = self.paths[:9800]
        # self.labels = self.labels[:9800]
        # self.size = len(self.paths)
        # opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        # obj_path = self.paths[index][1]
        label = self.paths[index][1]
        fname = self.paths[index][2]

        graph_list, label_dict = dgl.load_graphs(path)
        graph = graph_list[0]
        graph.create_formats_()

        # ms = pymeshlab.MeshSet()
        # ms.load_new_mesh(obj_path)
        # m = ms.current_mesh()
        # vertices = torch.from_numpy(m.vertex_matrix())
        # faces = torch.from_numpy(m.face_matrix())
        # ms.clear()
        # # mesh_faces = faces
        # if torch.equal(vertices, graph.ndata['pos']):
        #     mesh_faces = faces
        # else:
        #     print('node order does not match')
        #     print(fname)





        # graph.ndata['geometric_feat'][torch.isnan(graph.ndata['geometric_feat'])] = 0
        # graph.ndata['inv_feat'] = graph.ndata['inv_feat'][:, 1:]

        # if torch.sum(torch.isnan(graph.ndata['geometric_feat'])) != 0:
        #     print(path)
            # '/home/bs/Datasets/classification_datasets/ModelNet/Graphs_nonManifold_normalized_40/bed/train/bed_0199_simplified_to_1000_0_graph.bin'

        # assert torch.sum(torch.isnan(graph.ndata['geometric_feat'])) == 0, print(path)
        edge_length = graph.edata['edge_length']
        mid_point = 0.5 * (self.min_edge_length.item() + self.max_edge_length.item())
        graph_edge_length_bins = torch.cat((torch.linspace(self.min_edge_length.item(), mid_point, 24),
                                            torch.linspace(mid_point, self.max_edge_length.item(), 10)[1:])).\
            type(torch.float64)

        graph_edge_length_hist = torch.histogram(edge_length, bins=graph_edge_length_bins, density=True)[0]
        dihedral_angle = graph.edata['geometric_feat'][:, 1]
        graph_dihedral_angle_hist = torch.histogram(dihedral_angle, bins=32,
                                                    range=(self.min_dihedral_angle.item(), self.max_dihedral_angle.item())
                                                    , density=True)[0]
        graph_hist_feat = torch.cat((graph_edge_length_hist,graph_dihedral_angle_hist)).view(1, -1)
        # graph.ndata['inv_feat'] = graph.ndata['inv_feat'] [:, 1:]
        # ave_feat_4_5= torch.mean(graph.edata['geometric_feat'][:,-2:], 0)
        # std_feat_4_5 = torch.std(graph.edata['geometric_feat'][:,-2:], 0)
        # ave_4 = ave_feat_4_5[0]
        # ave_5 = ave_feat_4_5[1]
        # std_4 = std_feat_4_5[0]
        # std_5 = std_feat_4_5[1]
        # max = torch.max(graph.edata['geometric_feat'][:,-2:], 0)
        # graph.edata['geometric_feat'][:, -2: ] = graph.edata['geometric_feat'][:,-2:]/max.values[1]
        # graph.edata['geometric_feat'][:, 5] = graph.edata['geometric_feat'][:, 5] / max[1]
        # graph.edata['geometric_feat'][:,4] = (graph.edata['geometric_feat'][:,4] - ave_4) /(0.000001+std_4)
        # graph.edata['geometric_feat'][:,5] = (graph.edata['geometric_feat'][:,5] - ave_5) /(0.000001+std_5)

        # graph.apply_nodes(lambda nodes:
        #                   {'geometric_feat': (nodes.data['geometric_feat'] - self.mean_node_feat) / self.std_node_feat})
        # graph.apply_nodes(lambda nodes:
        #                   {'inv_feat': (nodes.data['inv_feat'] - self.mean_node_feat[6:]) / self.std_node_feat[6:]})
        # graph.apply_edges(lambda edges:
        #                   {'geometric_feat': (edges.data['geometric_feat'] - self.mean_edge_feat) / self.std_edge_feat})

        # graph.apply_nodes(lambda nodes:
        #                   {'pos': (nodes.data['pos'] - self.min_node_feat[0:3]) / (self.max_node_feat[0:3] - self.min_node_feat[0:3])})
        # graph.apply_nodes(lambda nodes:
        #                   {'normal': (nodes.data['normal'] - self.min_node_feat[3:6]) / (
        #                               self.max_node_feat[3:6] - self.min_node_feat[3:6])})
        # graph.apply_nodes(lambda nodes:
        #                   {'inv_feat': (nodes.data['inv_feat'] - self.min_node_feat[6:]) / (self.max_node_feat[6:] - self.min_node_feat[6:])})
        # graph.apply_edges(lambda edges:
        #                   {'geometric_feat': (edges.data['geometric_feat'] - self.min_edge_feat) /(self.max_edge_feat - self.min_edge_feat)})

        # if self.opt.save_features:
        #     return graph, graph_hist_feat, label, path
        # graph.ndata['inv_feat'] = graph.ndata['inv_feat'][:,2:]

        center = (torch.min(graph.ndata['pos'], dim=0)[0] + torch.max(graph.ndata['pos'], dim=0)[0]) * 0.5
        graph.ndata['pos'] = graph.ndata['pos'] - center
        max_len = torch.max(torch.linalg.norm(graph.ndata['pos'], dim=1))
        graph.ndata['pos'] /= max_len

        shape = graph_hist_feat.shape
        # graph_hist_feat = torch.ones(shape, dtype=torch.float64)
        return graph, graph_hist_feat, label, fname
        # return graph, graph_hist_feat, label, fname, mesh_faces

    def __len__(self):
        return self.size

    @staticmethod
    def find_classes(root):
        # df = pd.read_csv(dataset_info_csv)
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        labels = []
        obj = []
        dir = os.path.expanduser(dir)
        # dir_obj = dir.split('/')[1:-1] + ['Manifold40_manually_aligned_cleaned_translated_scaled_750F']
        # dir_obj = os.path.join(*dir_obj)
        # dir_obj = '/'+dir_obj
        for target in sorted(os.listdir(dir)):
            # phase = 'train' #Delete this
            d = os.path.join(dir, target, phase)
            # d_obj = os.path.join(dir_obj, target, phase)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_graph_file(fname) and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        # f_name_obj = fname.split('_')[:-2]
                        # f_name_obj = '_'.join(f_name_obj)+'.obj'
                        # obj_path = os.path.join(d_obj, f_name_obj)
                        # item = (path, obj_path, class_to_idx[target], fname)
                        item = (path, class_to_idx[target], fname)
                        labels.append(class_to_idx[target])
                        meshes.append(item)
            # phase = 'test'
            # d = os.path.join(dir, target, phase)
            # # d_obj = os.path.join(dir_obj, target, phase)
            # if not os.path.isdir(d):
            #     continue
            # for root, _, fnames in sorted(os.walk(d)):
            #     for fname in sorted(fnames):
            #         if is_graph_file(fname) and (root.count(phase) == 1):
            #             path = os.path.join(root, fname)
            #             # f_name_obj = fname.split('_')[:-2]
            #             # f_name_obj = '_'.join(f_name_obj)+'.obj'
            #             # obj_path = os.path.join(d_obj, f_name_obj)
            #             # item = (path, obj_path, class_to_idx[target], fname)
            #             item = (path, class_to_idx[target], fname)
            #             labels.append(class_to_idx[target])
            #             meshes.append(item)
        return meshes, labels

    @staticmethod
    def make_SHREC_S10_dataset_by_class(dir, class_to_idx):
        meshes = []
        labels = []
        obj = []
        dir = os.path.expanduser(dir)
        # dir_obj = dir.split('/')[1:-1] + ['Manifold40_manually_aligned_cleaned_translated_scaled_750F']
        # dir_obj = os.path.join(*dir_obj)
        # dir_obj = '/'+dir_obj
        for target in sorted(os.listdir(dir)):
            phase = 'train'
            d = os.path.join(dir, target, phase)
            # d_obj = os.path.join(dir_obj, target, phase)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_graph_file(fname) and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        # f_name_obj = fname.split('_')[:-2]
                        # f_name_obj = '_'.join(f_name_obj)+'.obj'
                        # obj_path = os.path.join(d_obj, f_name_obj)
                        # item = (path, obj_path, class_to_idx[target], fname)
                        item = (path, class_to_idx[target], fname)
                        labels.append(class_to_idx[target])
                        meshes.append(item)
            phase = 'test'
            d = os.path.join(dir, target, phase)
            # d_obj = os.path.join(dir_obj, target, phase)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_graph_file(fname) and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        # f_name_obj = fname.split('_')[:-2]
                        # f_name_obj = '_'.join(f_name_obj)+'.obj'
                        # obj_path = os.path.join(d_obj, f_name_obj)
                        # item = (path, obj_path, class_to_idx[target], fname)
                        item = (path, class_to_idx[target], fname)
                        labels.append(class_to_idx[target])
                        meshes.append(item)
        return meshes, labels
