import torch
import torch.nn as nn
from dgl.utils import expand_as_pair
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.nn.pytorch import GATConv
# from nn.collapse_8 import
from nn.collapse_4 import collapse_edge
from torch import linalg as LA
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_std
# import networkx as nx

class MeshConv(nn.Module):
    def __init__(self, node_in_feat, edge_in_feat, graph_in_feat, node_out_inv_feat,
                 edge_out_feat, graph_out_feat,sbin, batch_norm=False, edge_collapse=False, gnn_dropout=0.2):
        super(MeshConv, self).__init__()
        self.sbin = sbin
        self.node_out_inv_feat = node_out_inv_feat
        self.batch_norm = batch_norm
        self.linear_e = nn.Linear(2 * (node_in_feat - 6) + 2 + edge_in_feat + graph_in_feat, edge_out_feat, bias=False)
        self.f_e = nn.ReLU()
        self.e_dropout = nn.Dropout(gnn_dropout)
        self.linear_position = nn.Linear(edge_out_feat, 1, bias=False)
        self.f_position = nn.ReLU()
        self.linear_normal = nn.Linear(edge_out_feat, 1, bias=False)
        self.f_normal = nn.ReLU()
        self.linear_h_n = nn.Linear(node_in_feat - 6 + edge_out_feat + graph_in_feat, node_out_inv_feat, bias=False)
        self.f_h_n = nn.ReLU()
        self.h_n_dropout = nn.Dropout(gnn_dropout)
        self.linear_graph = nn.Linear(node_out_inv_feat + edge_out_feat + graph_in_feat, graph_out_feat, bias=False)
        self.f_g = nn.ReLU()
        self.g_dropout = nn.Dropout(gnn_dropout)
        self.linear_e_score = nn.Linear(edge_out_feat, 1, bias=False)
        self.node_pooling_linear = nn.Sequential(nn.Linear(self.sbin * node_out_inv_feat,2* node_out_inv_feat ), nn.ReLU(), nn.BatchNorm1d(node_out_inv_feat),
                                                 nn.Dropout(0.5), nn.Linear(2*node_out_inv_feat, node_out_inv_feat ), nn.ReLU(),
                                                 nn.BatchNorm1d(node_out_inv_feat),
                                                 nn.Dropout(0.5)
                                                 )
        if batch_norm:
            self.bn_node_pos = nn.BatchNorm1d(3)
            self.bn_node_normal = nn.BatchNorm1d(3)
            self.bn_node_inv_feat = nn.BatchNorm1d(node_out_inv_feat)
            self.bn_edge = nn.BatchNorm1d(edge_out_feat)
            # change to nn.BatchNomr1d(graph_out_feature) in case you ise batchsize>1
            self.bn_graph = nn.BatchNorm1d(graph_out_feat)
            # self.bn_graph = nn.InstanceNorm1d(graph_out_feat)

        self.edge_collapse = edge_collapse

        #Node encoding GAT layer
        self.gat_layer1 = GATConv(node_out_inv_feat, node_out_inv_feat,num_heads=2, attn_drop=0.4, residual= True, activation=F.elu)
        self.gat_layer2 = GATConv(2*node_out_inv_feat, int(node_out_inv_feat/2), num_heads=2, attn_drop=0.4, residual= True, activation=F.elu)

    def node_pooling(self, g, op):
        n_src = g.ndata['inv_feat']
        dim_size = g.batch_size * self.sbin
        if op == 'mean':
            node_pooled_feat = scatter_mean(n_src, g.ndata['node_bin_global'], dim=0, dim_size=dim_size)
        if op == 'sum':
            node_pooled_feat = scatter_add(n_src, g.ndata['node_bin_global'], dim=0, dim_size=dim_size)
        if op == 'max':
            node_pooled_feat, _ = scatter_max(n_src, g.ndata['node_bin_global'], dim=0, dim_size=dim_size)
        if op == 'std':
            node_pooled_feat = scatter_std(n_src, g.ndata['node_bin_global'], dim=0, dim_size=dim_size)

        # node_pooled_feat = node_pooled_feat.view(-1, self.node_out_inv_feat * self.sbin)
        return node_pooled_feat

    def node_encoder(self, g):
        cuda_ind = g.ndata['inv_feat'].get_device()
        cuda_device = 'cuda:'+str(cuda_ind)
        device = torch.device(cuda_device)
        node1 = torch.arange(g.batch_size * self.sbin, dtype=torch.int64, device=device).repeat_interleave(self.sbin)
        node2 = torch.squeeze(torch.arange(g.batch_size * self.sbin, dtype=torch.int64, device=device).view(-1, self.sbin).repeat(1,self.sbin).view(1,-1))

        g_bins = dgl.graph((node1, node2)).to(device)
        node_pooled_feat_mean = self.node_pooling(g, 'mean')
        g_bins.ndata['bin_mean'] = node_pooled_feat_mean
        bin_embedding = self.gat_layer1(g_bins, g_bins.ndata['bin_mean'])
        bin_embedding = bin_embedding.view(bin_embedding.shape[0], -1)
        bin_embedding = self.gat_layer2(g_bins, bin_embedding).view(bin_embedding.shape[0], -1)
        ind = torch.arange(g.batch_size, dtype=torch.int64, device=device).repeat_interleave(self.sbin)
        node_graph_embedding = scatter_mean(bin_embedding, ind, dim = 0)
        return node_graph_embedding

    def forward(self, g, graph_feat):
        cuda_ind = g.ndata['inv_feat'].get_device()
        cuda_device = 'cuda:'+str(cuda_ind)
        device = torch.device(cuda_device)
        # Update edge features using graph, relevant node and edge features
        g.apply_edges(fn.v_dot_u('normal', 'normal', 'cosine_sim'))
        g.apply_edges(fn.v_sub_u('pos', 'pos', 'pos_sub'))
        g.apply_edges(fn.v_sub_u('normal', 'normal', 'normal_sub'))
        # g.edata['uv_length'] = LA.vector_norm(g.edata['pos_sub'], dim=1)
        g.apply_edges(lambda edges: {'uv_length': LA.vector_norm(edges.data['pos_sub'], dim=1)})
        # g.apply_edges(lambda edges: {'uv_length': LA.vector_norm(edges.src['pos'] - edges.dst['pos'])})
        # a = torch.repeat_interleave(graph_feat, g.batch_num_edges(), dim=0)
        # print(a.is_cuda)
        g.apply_edges(lambda edges: {'temp': torch.cat((edges.src['inv_feat'], edges.dst['inv_feat'],
                                                        edges.data['uv_length'].view(-1, 1),
                                                        edges.data['cosine_sim'].view(-1, 1),
                                                        edges.data['geometric_feat'],
                                                        torch.repeat_interleave(graph_feat, g.batch_num_edges(), dim=0))
                                                       , 1)})
        # a = torch.repeat_interleave(graph_feat, g.batch_num_edges(), dim=0)
        g.edata['geometric_feat'] = self.f_e(self.linear_e(g.edata['temp']))

        # Update node features using relevant node and edge features
        # 1. Update node pos
        g.edata['pos_weight'] = self.f_position(self.linear_position(g.edata['geometric_feat']))
        g.edata['pos_update_e'] = g.edata['pos_sub'] * g.edata['pos_weight']
        # g.apply_edges(lambda edges: {'pos_update_e': edges.data['pos_sub'] * edges.data['pos_weight']})
        g.update_all(fn.copy_e('pos_update_e', 'hpos'), fn.mean('hpos', 'pos_update_n'))
        g.ndata['pos'] = g.ndata['pos'] + g.ndata['pos_update_n']

        # 2. Update node normal
        g.edata['normal_weight'] = self.f_normal(self.linear_normal(g.edata['geometric_feat']))
        g.edata['normal_update_e'] = g.edata['normal_sub'] * g.edata[
            'normal_weight']  # You can add (1-g.edata['cosine_sim'] as a coefficient
        g.update_all(fn.copy_e('normal_update_e', 'hnormal'), fn.mean('hnormal', 'normal_update_n'))
        g.ndata['normal'] = g.ndata['normal'] + g.ndata['normal_update_n']
        # Next line
        g.ndata['normal'] = g.ndata['normal'] / (LA.vector_norm(g.ndata['normal'], dim=1).view(-1, 1))
        # 3. Update node invariant features.
        g.update_all(fn.copy_e('geometric_feat', 'out'), fn.sum('out', 'agg_e'))
        g.ndata['inv_feat'] = self.f_h_n(self.linear_h_n(torch.cat((g.ndata['inv_feat'],
                                                                    g.ndata['agg_e'],
                                                                    torch.repeat_interleave(graph_feat,
                                                                                            g.batch_num_nodes(), dim=0))
                                                                   , 1)))
        # graph_feat = self.f_g(self.linear_graph(torch.cat((dgl.readout_nodes(g, 'inv_feat', op='mean'),
        #                                                    dgl.readout_edges(g, 'geometric_feat', op='mean'),
        #                                                    graph_feat), 1)))


        if self.batch_norm:
            g.ndata['pos'] = self.bn_node_pos(g.ndata['pos'])
            g.ndata['normal'] = self.bn_node_normal(g.ndata['normal'])
            g.ndata['inv_feat'] = self.bn_node_inv_feat(self.h_n_dropout(g.ndata['inv_feat']))
            g.edata['geometric_feat'] = self.bn_edge(self.e_dropout(g.edata['geometric_feat']))
            # graph_feat = self.bn_graph(graph_feat)

        g.edata['collapse_Score'] = self.compute_edge_score(g, 'softmax')

        if self.edge_collapse:
            g_new = collapse_edge(g, self.sbin, device)
            graph_feat = self.linear_graph(torch.cat((self.node_encoder(g_new),
                                                      dgl.readout_edges(g_new, 'geometric_feat', op='mean'),
                                                      graph_feat),1))
            graph_feat = self.g_dropout(self.f_g(graph_feat))
            if self.batch_norm:
                graph_feat = self.bn_graph(graph_feat)
        else:
            g_new = g
            graph_feat = self.linear_graph(torch.cat((self.node_encoder(g),
                                                      dgl.readout_edges(g, 'geometric_feat', op='mean'),
                                                      graph_feat),1))

            graph_feat = self.g_dropout(self.f_g(graph_feat))
            if self.batch_norm:
                graph_feat = self.bn_graph(graph_feat)

        return g_new, graph_feat

    def compute_edge_score(self, g, edgeScoreFunc):
        if edgeScoreFunc == 'softmax':
            edge_score_func = nn.Softmax()
        collapse_score = edge_score_func(self.linear_e_score(g.edata['geometric_feat']))
        return collapse_score


class ClassificationNetwork(nn.Module):

    def __init__(self, MeshConvLayers_spec, graph_classifier_FCN_spec, out_feat, graph_classifier_dropout, sbin, gnn_dropout=0.20):

        super(ClassificationNetwork, self).__init__()

        self.mesh_conv_layers = nn.ModuleList()
        # graph_out_feature=MeshConvLayers_spec[0][2]
        graph_out_feature = 64 #change to 64
        self.graph_out_feat_list = [64]
        for i in range(len(MeshConvLayers_spec)):
            node_in_feat, edge_in_feat, graph_in_feat, \
            node_out_inv_feat, edge_out_feat, graph_out_feat, batch_norm, edge_collapse = MeshConvLayers_spec[i]
            self.graph_out_feat_list.append(graph_out_feat)
            mesh_conv_layer = MeshConv(node_in_feat, edge_in_feat, graph_in_feat,
                                       node_out_inv_feat, edge_out_feat, graph_out_feat,sbin, batch_norm, edge_collapse, gnn_dropout)
            self.mesh_conv_layers.append(mesh_conv_layer)
            graph_out_feature = graph_out_feature + graph_out_feat
        # graph_classifier_FCN = [graph_out_feature] + graph_classifier_FCN_spec
        graph_classifier_FCN = [graph_out_feature] + graph_classifier_FCN_spec
        self.classify_graph_layers = nn.ModuleList()
        for i in range(len(graph_classifier_FCN) - 1):
            self.classify_graph_layers.append(nn.Linear(graph_classifier_FCN[i], graph_classifier_FCN[i + 1], bias=False))
            self.classify_graph_layers.append(nn.ReLU())
            self.classify_graph_layers.append(nn.Dropout(graph_classifier_dropout))
            self.classify_graph_layers.append(nn.BatchNorm1d(graph_classifier_FCN[i + 1]))

        self.classify_graph_layers.append(nn.Linear(graph_classifier_FCN[-1], out_feat))
        # self.classify_graph_layers.append(nn.ReLU())

        self.classify_graph = nn.Sequential(*self.classify_graph_layers)

        self.f_g = nn.ReLU()
        self.sbin = sbin
    def forward(self, g, g_feat):
        # g.ndata['geometric_feat'] = self.node_embedding(g.ndata['geometric_feat'])
        # g.edata['geometric_feat'] = self.edge_embedding(g.edata['geometric_feat'])
        g.ndata['init_pos'] = g.ndata['init_geometric_feat'][:, 0:3]
        cuda_ind = g_feat.get_device()
        cuda_device = 'cuda:'+str(cuda_ind)
        device = torch.device(cuda_device)
        layer = 0
        sbin = self.sbin
        # g_feature = []
        g.ndata['graph_ind'] = torch.ones(g.ndata['pos'].shape[0], device=device, dtype=torch.int64) * -1
        g.edata['graph_ind'] = torch.ones(g.edata['geometric_feat'].shape[0], device=device, dtype=torch.int64) * -1
        g.ndata['distance'] = torch.linalg.norm(g.ndata['pos'], dim=1)
        num_node_per_batch = g.batch_num_nodes()
        num_edge_per_batch = g.batch_num_edges()
        boundaries = torch.linspace(0, 1, sbin+1, device=device)
        node_bin = torch.bucketize(g.ndata['distance'], boundaries, right=False)
        node_bin[node_bin == 0] = 1
        node_bin[node_bin == sbin+1] = sbin
        g.ndata['node_bin'] = node_bin - 1

        # from dgl.backend import backend as F
        # F.segment_reduce()
        j=0
        i=0
        for k in range(g.batch_size):
            g.edata['graph_ind'][i:i+num_edge_per_batch[k]] = k
            g.ndata['graph_ind'][j:j+num_node_per_batch[k]] = k
            i = i + num_edge_per_batch[k]
            j = j + num_node_per_batch[k]

        g.ndata['node_bin_global'] = g.ndata['node_bin'] + sbin * g.ndata['graph_ind']

        g_feat_list = [g_feat]
        # g_feat_list = []
        for gnn in self.mesh_conv_layers:
            g, g_feat = gnn(g, g_feat)
            g_feat_list.append(g_feat)

        g_feat = torch.cat(g_feat_list,1)

        out = self.classify_graph(g_feat)

        return out