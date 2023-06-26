from collections import namedtuple
import torch
from torch import linalg as LA
from torch_scatter import scatter_add, scatter_mean, scatter, scatter_min, scatter_max
import dgl

def assign_bin_ind(g, sbin, device):
    # cuda_ind = 0
    # cuda_device = 'cuda:' + str(cuda_ind)
    # device = torch.device(cuda_device)
    g.ndata['graph_ind'] = torch.ones(g.ndata['init_pos'].shape[0], device=device, dtype=torch.int64) * -1
    g.edata['graph_ind'] = torch.ones(g.edata['geometric_feat'].shape[0], device=device, dtype=torch.int64) * -1
    g.ndata['distance'] = torch.linalg.norm(g.ndata['init_pos'], dim=1)
    # g.ndata['n_bin'] = torch.zeros([g.ndata['pos'].shape[0],1], device='cuda:0')
    num_node_per_batch = g.batch_num_nodes()
    num_edge_per_batch = g.batch_num_edges()
    # max_node_number = max(num_node_per_batch)
    # mesh_slice_info = torch.zeros([len(num_node_per_batch), max_node_number], device='cuda:0')
    # j = 0
    boundaries = torch.linspace(0, 1, sbin + 1, device=device)
    node_bin = torch.bucketize(g.ndata['distance'], boundaries, right=False)
    node_bin[node_bin == 0] = 1
    node_bin[node_bin == sbin + 1] = sbin
    g.ndata['node_bin'] = node_bin - 1

    # from dgl.backend import backend as F
    # F.segment_reduce()
    j = 0
    i = 0
    for k in range(g.batch_size):
        g.edata['graph_ind'][i:i + num_edge_per_batch[k]] = k
        g.ndata['graph_ind'][j:j + num_node_per_batch[k]] = k
        i = i + num_edge_per_batch[k]
        j = j + num_node_per_batch[k]

    g.ndata['node_bin_global'] = g.ndata['node_bin'] + sbin * g.ndata['graph_ind']
    return g

def collapse_edge(g, sbin, device):
    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch", "new_edge_score"])
    graph_list = dgl.unbatch(g)
    num_batches = len(graph_list)
    pooled_graph_feat_list=[]
    pooled_graph_list = []
    g_sample_ind = 0
    for g_sample in graph_list:
        g_sample = g_sample.to('cpu')
        nodes_remaining = set(range(g_sample.number_of_nodes()))
        # print(g_sample.number_of_edges())
        num_single_direction_edges = int(g_sample.number_of_edges() / 2)
        edge_remaining = set(range(num_single_direction_edges))
        cluster = torch.empty(g_sample.number_of_nodes(), device=torch.device('cpu'))  # device'cpu
        edge_score = torch.squeeze(g_sample.edata['collapse_Score'][:num_single_direction_edges])
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges
        i = 0
        k = 0
        j = 0
        new_edge_indices = []
        edge_index = torch.vstack([g_sample.edges()[0][:num_single_direction_edges],
                                   g_sample.edges()[1][:num_single_direction_edges]])
        if (torch.unique(edge_index, return_counts=True, dim=1)[1] > 1).sum().item() !=0:
            print('repeated')

        edge_index_cpu = edge_index.cpu()  # Change .cpu()
        edge_argsort_list = edge_argsort.tolist()
        edge_geometric_feat = g_sample.edata['geometric_feat'][:num_single_direction_edges]
        adjacent_planar_edges = g_sample.edata['adjacent_planar_edges'][:num_single_direction_edges].type(torch.int32)
        avoid_edge_collapse = []
        edge_cluster = torch.empty(edge_index.size()[1], device=torch.device('cpu'))

        for edge_idx in edge_argsort_list:

            if edge_idx in avoid_edge_collapse:
                continue

            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            source_adjacent_nodes = set(g_sample.successors(source).tolist())
            target_adjacent_nodes = set(g_sample.successors(target).tolist())
            source_target_common_node_neighbors = list(source_adjacent_nodes & target_adjacent_nodes)

            if len(source_target_common_node_neighbors) != 2:
                continue


            new_edge_indices.append(edge_idx)

            a, b, c, d = adjacent_planar_edges[edge_idx]
            edge_cluster[a] = k
            edge_cluster[b] = k
            edge_cluster[c] = k + 1
            edge_cluster[d] = k + 1
            edge_cluster[edge_idx] = -1
            try:
                edge_remaining.remove(a.item())
            except:
                if a.item() not in edge_remaining:
                    print(a.item(), 'does not exist')

            try:
                edge_remaining.remove(b.item())
            except:
                if b.item() not in edge_remaining:
                    print(b.item(), 'does not exist')

            try:
                edge_remaining.remove(c.item())
            except:
                if c.item() not in edge_remaining:
                    print(c.item(), 'does not exist')

            try:
                edge_remaining.remove(d.item())
            except:
                if d.item() not in edge_remaining:
                    print(d.item(), 'does not exist')

            try:
                edge_remaining.remove(edge_idx)
            except:
                if edge_idx not in edge_remaining:
                    print(edge_idx, 'does not exist')

            # edge_remaining.remove(b.item())
            # edge_remaining.remove(c.item())
            # edge_remaining.remove(d.item())
            # edge_remaining.remove(edge_idx)

            # merge adjacent planar edges
            ab_adjacent_edges = adjacent_planar_edges[a].tolist() + adjacent_planar_edges[b].tolist()
            ab_adjacent_planar_edges = torch.tensor(
                [i for i in ab_adjacent_edges if ((i != edge_idx) & (i != a.item()) & (i != b.item()))])
            cd_adjacent_edges = adjacent_planar_edges[c].tolist() + adjacent_planar_edges[d].tolist()
            cd_adjacent_planar_edges = torch.tensor(
                [i for i in cd_adjacent_edges if ((i != edge_idx) & (i != c.item()) & (i != d.item()))])
            avoid_edge_collapse = avoid_edge_collapse + ab_adjacent_planar_edges.tolist() + cd_adjacent_planar_edges.tolist()
            avoid_edge_collapse = list(set(avoid_edge_collapse))
            try:
                adjacent_planar_edges[a] = ab_adjacent_planar_edges
            except:
                print('adjacent_planar_edges[a]', adjacent_planar_edges[a])
                print('\n ab_adjacent_planar_edges', ab_adjacent_planar_edges)

            try:
                adjacent_planar_edges[b] = ab_adjacent_planar_edges
            except:
                print('adjacent_planar_edges[b]', adjacent_planar_edges[b])
                print('\n ab_adjacent_planar_edges', ab_adjacent_planar_edges)

            try:
                adjacent_planar_edges[c] = cd_adjacent_planar_edges
            except:
                print('adjacent_planar_edges[c]', adjacent_planar_edges[c])
                print('\n cd_adjacent_planar_edges', cd_adjacent_planar_edges)
            try:
                adjacent_planar_edges[d] = cd_adjacent_planar_edges
            except:
                print('adjacent_planar_edges[d]', adjacent_planar_edges[d])
                print('\n cd_adjacent_planar_edges', cd_adjacent_planar_edges)
            # print(ab_adjacent_planar_edges.size())
            # print(cd_adjacent_planar_edges.size())
            source_sord, source_adjacent_edges = torch.where(edge_index == source)
            source_adjacent_nodes = edge_index[(torch.vstack([1 - source_sord, source_adjacent_edges])).tolist()]
            target_sord, target_adjacent_edges = torch.where(edge_index == target)
            target_adjacent_nodes = edge_index[(torch.vstack([1 - target_sord, target_adjacent_edges])).tolist()]
        # source_target_common_node_neighbors = adjacent_planar_edges[edge_idx][4:]
            group_a_index = torch.where(source_adjacent_nodes != target)
            group_a = source_adjacent_nodes[group_a_index]
            group_b_index = torch.where(target_adjacent_nodes != source)
            group_b = target_adjacent_nodes[group_b_index]

            for m in group_a.tolist():
                for l in group_b.tolist():
                    t1 = torch.where((edge_index[0] == m) & (edge_index[1] == l))
                    t2 = torch.where((edge_index[1] == m) & (edge_index[0] == l))
                    if len(t1[0]) != 0:
                        if len(t1[0]) == 1:
                            avoid_edge_collapse.append(t1[0].item())
                        if len(t1[0]) > 1:
                            print(t1)

                    else:
                        if len(t2[0]) != 0:
                            if len(t2[0]) == 1:
                                avoid_edge_collapse.append(t2[0].item())
                            if len(t2[0]) > 1:
                                print(t2)

            cluster[source] = i
            nodes_remaining.remove(source)
            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)
            i += 1
            k += 2

    # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1

    # The remaining edges are simply kept.
        for edge_id in edge_remaining:
            edge_cluster[edge_id] = k
            k += 1

        edge_cluster = edge_cluster.type(torch.int64)
        edge_cluster_w_o_collapsed_edges = torch.where(edge_cluster == -1, k, edge_cluster)
        edge_cluster_w_o_collapsed_edges = edge_cluster_w_o_collapsed_edges.type(torch.int64)
        edge_new_geometric_feat = scatter_mean(edge_geometric_feat, edge_cluster_w_o_collapsed_edges, dim=0)
        edge_new_geometric_feat = edge_new_geometric_feat[:-1]
        edge_new_adjacent_planar_edges, argmin = scatter_min(adjacent_planar_edges, edge_cluster_w_o_collapsed_edges, dim=0)
        edge_new_adjacent_planar_edges = edge_new_adjacent_planar_edges[:-1]
        edge_new_adjacent_planar_edges_max, argmax = scatter_max(adjacent_planar_edges, edge_cluster_w_o_collapsed_edges, dim=0)
        edge_new_adjacent_planar_edges_max = edge_new_adjacent_planar_edges_max[:-1]
        if (edge_new_adjacent_planar_edges != edge_new_adjacent_planar_edges_max).sum()>0:
            print('not matched!')
        # print(edge_cluster.shape[0])
        edge_new_adjacent_planar_edges_renamed = edge_cluster[edge_new_adjacent_planar_edges.type(torch.int64)]

        cluster = cluster.type(torch.int64)
        node_new_inv_feat = scatter_add(g_sample.ndata['inv_feat'], cluster, dim=0, dim_size=i)
        node_new_pos = scatter_mean(g_sample.ndata['pos'], cluster, dim=0, dim_size=i)
        node_init_pos = scatter_mean(g_sample.ndata['init_pos'], cluster, dim=0, dim_size=i)
        node_new_normal= scatter_mean(g_sample.ndata['normal'], cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]

        if len(nodes_remaining):
            remaining_score = g_sample.ndata['inv_feat'].new_ones \
                ((node_new_inv_feat.size(0) - len(new_edge_indices),))
            new_edge_score = torch.cat([new_edge_score, remaining_score])

        node_new_inv_feat = node_new_inv_feat * \
                              new_edge_score.view(-1, 1)

        N = node_new_inv_feat.size(0)
        # print(cluster[edge_index])
        if ((torch.unique(cluster[edge_index], dim=1, return_counts=True)[1] > 2) * 1).sum().item() != 0:
            print('Not expected to have more than 2')

        new_edge_index_temp, _ = torch.sort(cluster[edge_index], dim=0)
        new_edge_index_mymethod, arg_min = scatter_min(torch.transpose(new_edge_index_temp, 0, 1),
                                                   edge_cluster_w_o_collapsed_edges, dim=0)
        new_edge_index_mymethod = new_edge_index_mymethod[:-1]
        new_edge_index_mymethod = torch.transpose(new_edge_index_mymethod, 0, 1)
        new_edge_index_opposite_direction = torch.vstack([new_edge_index_mymethod[1], new_edge_index_mymethod[0]])
        new_edge_index = torch.hstack([new_edge_index_mymethod, new_edge_index_opposite_direction])
        new_edge_feature_ = torch.vstack([edge_new_geometric_feat, edge_new_geometric_feat])
        new_adjacent_planar_edges0 = torch.vstack(
            [edge_new_adjacent_planar_edges_renamed, edge_new_adjacent_planar_edges_renamed])

        g_new = dgl.graph((new_edge_index[0], new_edge_index[1]), num_nodes=N, device=device)
        if (torch.unique(new_edge_index, return_counts=True, dim=1)[1] > 1).sum().item() != 0:
            print('repeated')
        g_new.ndata['inv_feat'] = node_new_inv_feat.to(device)
        g_new.ndata['pos'] = node_new_pos.to(device)
        g_new.ndata['init_pos'] = node_init_pos.to(device)
        g_new.ndata['normal'] = node_new_normal.to(device)
        g_new.ndata['normal'] = g_new.ndata['normal'] / (LA.vector_norm(g_new.ndata['normal'], dim=1).view(-1, 1))
        g_new.edata['geometric_feat'] = new_edge_feature_.to(device)
        g_new.edata['adjacent_planar_edges'] = new_adjacent_planar_edges0.to(device)
        # g_new_feat = f_g(linear_graph(torch.cat((dgl.readout_nodes(g_new, 'inv_feat', op='mean'),
        #                                          dgl.readout_edges(g_new, 'geometric_feat', op='mean'),
        #                                          graph_feat[g_sample_ind, :].view(1,-1)), 1)))
        # pooled_graph_feat_list.append(g_new_feat)
        pooled_graph_list.append(g_new)
        g_sample_ind = g_sample_ind+1
    # num_nodes_in_batch = torch.bincount(new_nodes_batch_id)
    # g_new.set_batch_num_nodes(num_nodes_in_batch)
    pooled_g_batch = dgl.batch(pooled_graph_list)
    pooled_g_batch = assign_bin_ind(pooled_g_batch, sbin, device)
    # graph_feat = torch.cat(pooled_graph_feat_list, dim=0)

    return pooled_g_batch
