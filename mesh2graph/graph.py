import numpy as np
import pymesh
import torch
import dgl


class GraphConstructor:

    def __init__(self, input_mesh, mesh_name, filename_to_save, graph_label):
        self.__graph = None
        self.__input_mesh = input_mesh
        self.__mesh_name = mesh_name
        self.__input_mesh.enable_connectivity()
        self.__mesh_nodes, self.__mesh_edges = pymesh.mesh_to_graph(self.__input_mesh)
        self.__mesh_faces = self.__input_mesh.faces
        self.__filename_to_save = filename_to_save
        self.__graph_label = graph_label
        self.two_face_neighbor = True

    def create_graphs(self):


        # node Features
        self.__input_mesh.add_attribute("vertex_valance")
        self.__input_mesh.add_attribute("vertex_normal")
        self.__input_mesh.add_attribute("vertex_area")
        self.__input_mesh.add_attribute("vertex_mean_curvature")
        self.__input_mesh.add_attribute("vertex_gaussian_curvature")
        self.__input_mesh.add_attribute("vertex_dihedral_angle")

        node_features = np.concatenate((self.__mesh_nodes, self.__input_mesh.get_vertex_attribute("vertex_normal"),
                                        self.__input_mesh.get_vertex_attribute("vertex_mean_curvature"),
                                        self.__input_mesh.get_vertex_attribute("vertex_gaussian_curvature"),
                                        self.__input_mesh.get_vertex_attribute("vertex_area"),
                                        self.__input_mesh.get_vertex_attribute("vertex_dihedral_angle")), axis=1)
        nodes_coordinate = self.__mesh_nodes.copy()

        # face_features
        self.__input_mesh.add_attribute("face_normal")
        self.__input_mesh.add_attribute("face_area")
        face_normal = self.__input_mesh.get_attribute("face_normal").reshape((self.__mesh_faces.shape[0], 3))
        face_area = self.__input_mesh.get_attribute("face_area").reshape((self.__mesh_faces.shape[0], 1))

        # edge feature
        edge_features = np.zeros((self.__mesh_edges.shape[0], 6))
        edge_adjacent_edges_snp = np.zeros(((self.__mesh_edges.shape[0], 4)))
        for edge_idx in range(self.__mesh_edges.shape[0]):
            edge_features[edge_idx] = self.get_edge_feature(edge_idx, face_normal, face_area)
            edge_adjacent_edges_snp[edge_idx] = self.get_edge_adjacent_edges_and_oposite_nodes(edge_idx)
            if not (self.two_face_neighbor):
                break
            # edge_labels[edge_idx], edge_soft_labels[edge_idx], edge_areas[edge_idx] = self.get_edge_label(edge_idx)
        if not (self.two_face_neighbor):
            return True

        # get nodes adjacent to nodes
        nodes_nodes_adjacency = self.get_vertices_influenced_by_vertices()
        edge_features_bidirectional = np.concatenate((edge_features[:, 1:], edge_features[:, 1:]), axis=0)
        edge_length_bidirectional = np.concatenate((edge_features[:, 0], edge_features[:, 0]), axis=0)
        edge_adjacent_edges_snp_bidirectional = np.concatenate((edge_adjacent_edges_snp, edge_adjacent_edges_snp))
        src = torch.tensor(nodes_nodes_adjacency[:, 0])
        dst = torch.tensor(nodes_nodes_adjacency[:, 1])
        graph = dgl.graph((src, dst))
        graph.ndata['init_geometric_feat'] = torch.tensor(node_features)
        graph.ndata['pos'] = torch.tensor(node_features[:, 0:3])
        graph.ndata['normal'] = torch.tensor(node_features[:, 3:6])
        graph.ndata['inv_feat'] = torch.tensor(node_features[:, 6:])
        graph.ndata['geometric_feat'] = torch.tensor(node_features)
        graph.edata['init_geometric_feat'] = torch.tensor(edge_features_bidirectional)
        graph.edata['geometric_feat'] = torch.tensor(edge_features_bidirectional)
        graph.edata['init_edge_length'] = torch.tensor(edge_length_bidirectional)
        graph.edata['edge_length'] = torch.tensor(edge_length_bidirectional)
        graph.edata['adjacent_planar_edges'] = torch.tensor(edge_adjacent_edges_snp_bidirectional)
        # graph = dgl.to_bidirected(graph, copy_ndata=True)
        dgl.data.utils.save_graphs(self.__filename_to_save, [graph], self.__graph_label)
        return graph

    def get_edge_feature(self, edge_idx, face_normal, face_area):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        n_0_adjacent_face_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_0))
        n_1_adjacent_faces_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_1))
        edge_adjacent_faces_indices = np.sort(list(n_0_adjacent_face_indices & n_1_adjacent_faces_indices))
        if len(edge_adjacent_faces_indices) == 2:
            face_idx_0 = edge_adjacent_faces_indices[0]
            face_idx_1 = edge_adjacent_faces_indices[1]
        if len(edge_adjacent_faces_indices) != 2:
            print(f"mesh {self.__mesh_name} : edge {edge_idx} has {len(edge_adjacent_faces_indices)} adjacent face.")
            self.two_face_neighbor = False
            return True
        # get dihedral angle

        face_0_normal = face_normal[face_idx_0]
        face_1_normal = face_normal[face_idx_1]
        cos_theta = min(np.dot(face_0_normal, face_1_normal), 1)
        cos_theta = max(-1, cos_theta)
        dihedral_angle = np.expand_dims(np.pi - np.arccos(cos_theta), axis=0)

        # get edge height ratio
        edge_norm = np.expand_dims(np.linalg.norm(self.__mesh_nodes[n_0] - self.__mesh_nodes[n_1]), axis=0)
        face_0_area = face_area[face_idx_0]
        face_1_area = face_area[face_idx_1]
        edge_height_ratios = edge_norm ** 2 / np.array([2 * face_0_area, 2 * face_1_area])
        edge_height_ratios = np.squeeze(edge_height_ratios)

        # get opposite angle
        opposite_vertex_in_face_idx_0 = self.__mesh_nodes[list(set(self.__mesh_faces[face_idx_0]) -
                                                               set(self.__mesh_edges[edge_idx]))[0]]
        opposite_vertex_in_face_idx_1 = self.__mesh_nodes[list(set(self.__mesh_faces[face_idx_1]) -
                                                               set(self.__mesh_edges[edge_idx]))[0]]
        edge_a = self.__mesh_nodes[n_0] - opposite_vertex_in_face_idx_0
        edge_a = edge_a / np.linalg.norm(edge_a)
        edge_b = self.__mesh_nodes[n_1] - opposite_vertex_in_face_idx_0
        edge_b = edge_b / np.linalg.norm(edge_b)
        gamma_1 = np.arccos(np.dot(edge_a, edge_b))

        edge_c = self.__mesh_nodes[n_0] - opposite_vertex_in_face_idx_1
        edge_c = edge_c / np.linalg.norm(edge_c)
        edge_d = self.__mesh_nodes[n_1] - opposite_vertex_in_face_idx_1
        edge_d = edge_d / np.linalg.norm(edge_d)
        gamma_2 = np.arccos(np.dot(edge_c, edge_d))
        opposite_angles = np.array([gamma_1, gamma_2])

        return np.concatenate((edge_norm, dihedral_angle, opposite_angles, edge_height_ratios))

    def get_vertices_influenced_by_vertices(self):
        nodes_nodes_adjacency = np.concatenate((self.__mesh_edges, self.__mesh_edges[:, ::-1]), axis=0)
        return nodes_nodes_adjacency

    def get_edge_adjacent_edges_and_oposite_nodes(self, edge_idx):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        all_adjacent_edges = (set(self.get_vertex_adjacent_edges(n_0)) | set(self.get_vertex_adjacent_edges(n_1))) - \
                             {edge_idx}
        edge_adjacent_faces = self.get_edge_adjacent_faces(edge_idx)
        assert (len(edge_adjacent_faces) ==2), f" Edge {edge_idx} in {self.__mesh_name} has more than two adjacent face"
        face_idx_0 = edge_adjacent_faces[0]
        face_idx_1 = edge_adjacent_faces[1]

        # find opposite nodes of the edge_idx
        n_2 = list(set(self.__mesh_faces[face_idx_0]) - set(self.__mesh_edges[edge_idx]))[0]
        n_3 = list(set(self.__mesh_faces[face_idx_1]) - set(self.__mesh_edges[edge_idx]))[0]

        adjacent_edges_share_node_plane = []
        assert (not (n_0 == n_2)), f"The mesh {self.__mesh_name} is not two manifold"
        if n_0 < n_2:
            e_share_n_p_idx_0 = np.where((self.__mesh_edges[:, 0] == n_0) & (self.__mesh_edges[:, 1] == n_2))[0]
        else:
            e_share_n_p_idx_0 = np.where((self.__mesh_edges[:, 0] == n_2) & (self.__mesh_edges[:, 1] == n_0))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_0[0])

        assert (n_1 != n_2), f"The mesh {self.__mesh_name} is not two manifold"
        if n_1 < n_2:
            e_share_n_p_idx_2 = np.where((self.__mesh_edges[:, 0] == n_1) & (self.__mesh_edges[:, 1] == n_2))[0]
        else:
            e_share_n_p_idx_2 = np.where((self.__mesh_edges[:, 0] == n_2) & (self.__mesh_edges[:, 1] == n_1))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_2[0])

        assert (n_0 != n_3), f"The mesh {self.__mesh_name} is not two manifold"
        if n_0 < n_3:
            e_share_n_p_idx_1 = np.where((self.__mesh_edges[:, 0] == n_0) & (self.__mesh_edges[:, 1] == n_3))[0]
        else:
            e_share_n_p_idx_1 = np.where((self.__mesh_edges[:, 0] == n_3) & (self.__mesh_edges[:, 1] == n_0))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_1[0])

        assert (n_1 != n_3), f"The mesh {self.__mesh_name} is not two manifold"
        if n_1 < n_3:
            e_share_n_p_idx_3 = np.where((self.__mesh_edges[:, 0] == n_1) & (self.__mesh_edges[:, 1] == n_3))[0]
        else:
            e_share_n_p_idx_3 = np.where((self.__mesh_edges[:, 0] == n_3) & (self.__mesh_edges[:, 1] == n_1))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_3[0])

        # adjacent_edges_share_only_node = np.array(list(all_adjacent_edges - set(adjacent_edges_share_node_plane)))

        return np.array(adjacent_edges_share_node_plane)

    def get_vertex_adjacent_edges(self, n_idx):
        return np.where((self.__mesh_edges[:, 0] == n_idx) | (self.__mesh_edges[:, 1] == n_idx))[0]

    def get_edge_adjacent_faces(self, edge_idx):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        n_0_adjacent_face_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_0))
        n_1_adjacent_faces_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_1))
        edge_adjacent_faces_indices = np.array(list(n_0_adjacent_face_indices & n_1_adjacent_faces_indices))
        return edge_adjacent_faces_indices
