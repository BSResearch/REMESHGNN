import torch.utils.data as data
import numpy as np
import pickle
import os
import dgl
import torch
from util.util import is_graph_file
import pandas as pd

class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        super(BaseDataset, self).__init__()

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
        num_samples_per_class_cache = os.path.join(self.root, 'num_samples_per_class_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')

            # read the first graph to get the dimensions
            sample_target = os.listdir(self.root)[0]
            d = os.path.join(self.root, sample_target, 'train')
            for root, _, fnames in sorted(os.walk(d)):
                for i in range(1):
                    g_sample_name = fnames[i]
                    g_sample_path = os.path.join(d, g_sample_name)
                break

            g_sample, _ = dgl.load_graphs(g_sample_path)
            g_sample = g_sample[0]
            node_feat_len = g_sample.ndata['geometric_feat'].shape[1]
            edge_feat_len = g_sample.edata['geometric_feat'].shape[1]
            mean_node_feat, std_node_feat = torch.zeros((node_feat_len)), torch.zeros((node_feat_len))
            mean_edge_feat, std_edge_feat = torch.zeros((edge_feat_len)), torch.zeros((edge_feat_len))
            count = 0
            edge_length_min = g_sample.edata['edge_length'].min(axis=0)
            edge_length_max = g_sample.edata['edge_length'].max(axis=0)
            dihedral_angle_min = g_sample.edata['geometric_feat'][:, 1].min(axis=0)
            dihedral_angle_max = g_sample.edata['geometric_feat'][:, 1].max(axis=0)
            node_feat_min = torch.min(g_sample.ndata['geometric_feat'], 0).values
            node_feat_max = torch.max(g_sample.ndata['geometric_feat'], 0).values
            edge_feat_min = torch.min(g_sample.edata['geometric_feat'], 0).values
            edge_feat_max = torch.max(g_sample.edata['geometric_feat'], 0).values
            # num_samples_per_class_dict= {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0,
            #                              11:0, 12:0, 13:0, 14:0, 15:0}
            num_samples_per_class_dict= {i:0 for i in range(self.nclasses)}
            for target in sorted(os.listdir(self.root)):
                d = os.path.join(self.root, target, 'train')
                num_samples_per_class_dict[self.class_to_idx[target]] = len(os.listdir(d))
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if is_graph_file(fname):
                            fpath = os.path.join(root, fname)
                            g_list, label_dict = dgl.load_graphs(fpath)
                            g=g_list[0]


                            # node features
                            node_feat_mean_single_graph = g.ndata['geometric_feat'].mean(axis=0)
                            node_feat_std_single_graph = g.ndata['geometric_feat'].std(axis=0)
                            node_feat_min_single_graph = torch.min(g.ndata['geometric_feat'], 0).values
                            node_feat_max_single_graph = torch.max(g.ndata['geometric_feat'], 0).values
                            mean_node_feat = mean_node_feat + node_feat_mean_single_graph
                            std_node_feat = std_node_feat + node_feat_std_single_graph
                            node_temp_min = torch.cat((node_feat_min.view(1,-1), node_feat_min_single_graph.view(1,-1)), 0)
                            node_feat_min = torch.min(node_temp_min, 0).values
                            node_temp_max = torch.cat((node_feat_max.view(1,-1),node_feat_max_single_graph.view(1,-1)), 0)
                            node_feat_max = torch.max(node_temp_max, 0).values
                            # edge features
                            edge_feat_mean_single_graph = g.edata['geometric_feat'].mean(axis=0)
                            edge_feat_std_single_graph = g.edata['geometric_feat'].std(axis=0)
                            edge_feat_min_single_graph = torch.min(g.edata['geometric_feat'], 0).values
                            edge_feat_max_single_graph = torch.max(g.edata['geometric_feat'], 0).values
                            mean_edge_feat = mean_edge_feat + edge_feat_mean_single_graph
                            std_edge_feat = std_edge_feat + edge_feat_std_single_graph
                            edge_temp_min = torch.cat((edge_feat_min.view(1,-1), edge_feat_min_single_graph.view(1,-1)), 0)
                            edge_feat_min = torch.min(edge_temp_min,0).values
                            edge_temp_max = torch.cat((edge_feat_max.view(1,-1), edge_feat_max_single_graph.view(1,-1)), 0)
                            edge_feat_max = torch.max(edge_temp_max, 0).values

                            count = count + 1
                            g_edge_length_min = g.edata['edge_length'].min(axis=0)
                            g_edge_length_max = g.edata['edge_length'].max(axis=0)
                            g_dihedral_angle_min = g.edata['geometric_feat'][:, 1].min(axis=0)
                            g_dihedral_angle_max = g.edata['geometric_feat'][:, 1].max(axis=0)

                            if g_edge_length_min < edge_length_min:
                                edge_length_min = g_edge_length_min

                            if g_edge_length_max > edge_length_max:
                                edge_length_max = g_edge_length_max

                            if g_dihedral_angle_min <  dihedral_angle_min:
                                dihedral_angle_min = g_dihedral_angle_min

                            if g_dihedral_angle_max > dihedral_angle_max:
                                dihedral_angle_max = g_dihedral_angle_max


            mean_node_feat = mean_node_feat / count
            std_node_feat = std_node_feat / count
            mean_edge_feat = mean_edge_feat / count
            std_edge_feat = std_edge_feat / count

            transform_dict = {
                'mean_node_feat': mean_node_feat,
                'std_node_feat': std_node_feat,
                'node_feat_len': node_feat_len,
                'min_node_feat': node_feat_min,
                'max_node_feat': node_feat_max,
                'mean_edge_feat': mean_edge_feat,
                'std_edge_feat': std_edge_feat,
                'edge_feat_len': edge_feat_len,
                'min_edge_feat': edge_feat_min,
                'max_edge_feat': edge_feat_max,
                'min_edge_length': edge_length_min.values,
                'max_edge_length': edge_length_max.values,
                'min_dihedral_angle': dihedral_angle_min.values,
                'max_dihedral_angle': dihedral_angle_max.values

            }
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)

            with open(num_samples_per_class_cache, 'wb') as f2:
                pickle.dump(num_samples_per_class_dict, f2)
            print('saved: ', num_samples_per_class_cache)


        # open mean/std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean_node_feat = transform_dict['mean_node_feat']
            self.std_node_feat = transform_dict['std_node_feat']
            self.node_feat_len = transform_dict['node_feat_len']
            self.min_node_feat = transform_dict['min_node_feat']
            self.max_node_feat = transform_dict['max_node_feat']
            self.mean_edge_feat = transform_dict['mean_edge_feat']
            self.std_edge_feat = transform_dict['std_edge_feat']
            self.edge_feat_len = transform_dict['edge_feat_len']
            self.min_edge_feat = transform_dict['min_edge_feat']
            self.max_edge_feat = transform_dict['max_edge_feat']
            self.min_edge_length = transform_dict['min_edge_length']
            self.max_edge_length = transform_dict['max_edge_length']
            self.min_dihedral_angle = transform_dict['min_dihedral_angle']
            self.max_dihedral_angle = transform_dict['max_dihedral_angle']

        with open(num_samples_per_class_cache,'rb') as f2:
            self.num_samples_per_class_dict = pickle.load(f2)

            print('loaded number of samples per class from cache')

def collate_fn(samples):
    # use for classification
    # graphs, graph_feat, labels, fnames, mesh_faces = map(list, zip(*samples))
    graphs, graph_feat, labels, fnames = map(list, zip(*samples))
    meta_labels = {}
    num_nodes_cum = 0

    # for i in range(len(graphs)):
    #     mesh_faces[i] = mesh_faces[i] + num_nodes_cum
    #     num_nodes_cum = num_nodes_cum + graphs[i].num_nodes()

    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(graph_feat, dim=0), torch.tensor(labels), fnames
    # return batched_graph, torch.cat(graph_feat, dim=0), torch.tensor(labels), fnames, torch.cat(mesh_faces, dim=0)


def collate_fn_2(samples):
    #use for test classification
    graphs, graph_feat, labels, path = map(list, zip(*samples))
    meta_labels = {}
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(graph_feat, dim=0), torch.tensor(labels), path

def collate_fn_8(samples):
    #use for test classification
    graphs, graph_feat, path = map(list, zip(*samples))
    meta_labels = {}
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(graph_feat, dim=0),  path

def collate_fn_3(samples):
    #use for LTR
    query_graph, query_graph_hist_feat, candidate_graph_batched, candidate_hist_feats, distance = map(list, zip(*samples))
    meta_labels = {}
    # keys = labels[0].keys()
    batched_query_graph = dgl.batch(query_graph)
    batched_candidate_graph = dgl.batch(candidate_graph_batched)
    # for key in keys:
    #     meta_labels.update({key: torch.cat([label[key] for label in labels])})
    return batched_query_graph, torch.cat(query_graph_hist_feat, dim=0), batched_candidate_graph, \
           torch.cat(candidate_hist_feats, dim=0), torch.cat(distance, dim=0)

def collate_fn_4(samples):
    #use for retrieval test
    query_graph, query_graph_hist_feat, path = map(list, zip(*samples))
    batched_query_graph = dgl.batch(query_graph)
    return batched_query_graph, torch.cat(query_graph_hist_feat, dim=0), path

def collate_fn_5(samples):
    query_graph, query_graph_hist_feat, candidate_graph_batched, candidate_hist_feats, distance = map(list, zip(*samples))
    batched_query_graph = dgl.batch(query_graph)
    batched_candidate_graph = dgl.batch(candidate_graph_batched)

    return batched_query_graph, torch.cat(query_graph_hist_feat, dim=0), batched_candidate_graph, \
           torch.cat(candidate_hist_feats, dim=0), torch.cat(distance, dim=0)


def collate_fn_6(samples):
    # use for positive negative pair retrieval
    query_graph, query_graph_hist_feat, \
    pos_candidate_graph_batched, pos_candidate_hist_feats, \
    neg_candidate_graph_batched, neg_candidate_hist_feats = map(list, zip(*samples))
    batched_query_graph = dgl.batch(query_graph)
    batched_pos_candidate_graph = dgl.batch(pos_candidate_graph_batched)
    batched_neg_candidate_graph = dgl.batch(neg_candidate_graph_batched)
    return batched_query_graph, torch.cat(query_graph_hist_feat, dim=0),\
           batched_pos_candidate_graph, torch.cat(pos_candidate_hist_feats, dim=0),\
           batched_neg_candidate_graph, torch.cat(neg_candidate_hist_feats, dim=0)

def collate_fn_7(samples):
    # use for positive negative pair retrieval
    query_graph, query_graph_hist_feat, \
    pos_candidate_graph_batched, pos_candidate_hist_feats, \
    neg_candidate_graph_batched, neg_candidate_hist_feats = map(list, zip(*samples))
    all_graph_list = []
    all_feat_list = []
    for i in range(len(query_graph)):
        all_graph_list.append(query_graph[i])
        all_feat_list.append(query_graph_hist_feat[i])
    for i in range(len(query_graph)):
        all_graph_list.append(pos_candidate_graph_batched[i])
        all_feat_list.append(pos_candidate_hist_feats[i])
    for i in range(len(query_graph)):
        all_graph_list.append(neg_candidate_graph_batched[i])
        all_feat_list.append(neg_candidate_hist_feats[i])
    all_graph = dgl.batch(all_graph_list)
    # all_feat = batched_query_graph + pos_candidate_hist_feats + neg_candidate_hist_feats
    all_feat = torch.cat(all_feat_list)
    # return batched_query_graph, torch.cat(query_graph_hist_feat, dim=0),\
    #        batched_pos_candidate_graph, torch.cat(pos_candidate_hist_feats, dim=0),\
    #        batched_neg_candidate_graph, torch.cat(neg_candidate_hist_feats, dim=0)
    return all_graph, all_feat
