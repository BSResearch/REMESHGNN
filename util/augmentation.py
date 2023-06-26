import numpy as np
import pymesh


def augment_mesh(mesh, mean, var):
    return scale_verts(mesh, mean, var)


def scale_verts(mesh, mean, var, coef):
    vertices = mesh.vertices.copy()

    for vertex in vertices:
        vertex += coef * np.random.normal(mean, var)

    mesh = pymesh.form_mesh(vertices, mesh.faces)

    return mesh


def rotateX3D(input_mesh, theta):
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), np.sin(theta)],
                                [0, -np.sin(theta), np.cos(theta)]])
    vertices = input_mesh.vertices.copy()
    vertices = np.matmul(vertices, rotation_matrix)
    rotated_mesh = pymesh.form_mesh(vertices, input_mesh.faces)
    return rotated_mesh


def rotateY3D(input_mesh, theta):
    rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-np.sin(theta), 0, np.cos(theta)]])

    vertices = input_mesh.vertices.copy()
    vertices = np.matmul(vertices, rotation_matrix)
    rotated_mesh = pymesh.form_mesh(vertices, input_mesh.faces)
    return rotated_mesh


def rotateZ3D(input_mesh, theta):
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    vertices = input_mesh.vertices.copy()
    vertices = np.matmul(vertices, rotation_matrix)
    rotated_mesh = pymesh.form_mesh(vertices, input_mesh.faces)
    return rotated_mesh


def slide_verts(input_mesh, prct, mesh_name):
    dihedral_angle_list = []
    mesh_nodes, mesh_edges = pymesh.mesh_to_graph(input_mesh)
    mesh_faces = input_mesh.faces
    input_mesh.add_attribute("face_normal")
    face_normal = input_mesh.get_attribute("face_normal").reshape((mesh_faces.shape[0], 3))
    all_edges_has_two_faces = True
    vertices = input_mesh.vertices.copy()
    for edge_idx in range(len(mesh_edges)):
        n_0 = mesh_edges[edge_idx, 0]
        n_1 = mesh_edges[edge_idx, 1]
        n_0_adjacent_face_indices = set(input_mesh.get_vertex_adjacent_faces(n_0))
        n_1_adjacent_faces_indices = set(input_mesh.get_vertex_adjacent_faces(n_1))
        edge_adjacent_faces_indices = np.sort(list(n_0_adjacent_face_indices & n_1_adjacent_faces_indices))
        if len(edge_adjacent_faces_indices) == 2:
            face_idx_0 = edge_adjacent_faces_indices[0]
            face_idx_1 = edge_adjacent_faces_indices[1]

        if len(edge_adjacent_faces_indices) != 2:
            print(
                f"message generated in slide function. mesh {mesh_name} : edge {edge_idx} has {len(edge_adjacent_faces_indices)} adjacent face.")
            all_edges_has_two_faces = False
            break

        face_0_normal = face_normal[face_idx_0]
        face_1_normal = face_normal[face_idx_1]
        cos_theta = min(np.dot(face_0_normal, face_1_normal), 1)
        cos_theta = max(-1, cos_theta)
        dihedral_angle = np.expand_dims(np.pi - np.arccos(cos_theta), axis=0)
        dihedral_angle_list.append(dihedral_angle)

    if all_edges_has_two_faces:
        dihedral_angle_array = np.array(dihedral_angle_list)

        vids = np.random.permutation(len(vertices))
        target = int(prct * len(vids))
        shifted = 0
        for vi in vids:
            if shifted < target:
                edges = np.where((mesh_edges[:, 0] == vi) | (mesh_edges[:, 1] == vi))[0]
                if min(dihedral_angle_array[edges]) > 2.65:
                    edge = mesh_edges[np.random.choice(edges)]
                    vi_t = edge[1] if vi == edge[0] else edge[0]
                    nv = vertices[vi] + np.random.uniform(0.2, 0.5) * (vertices[vi_t] - vertices[vi])
                    vertices[vi] = nv
                    shifted += 1
            else:
                break
        shifted = shifted / len(vertices)

    slided_mesh = pymesh.form_mesh(vertices, input_mesh.faces)

    return all_edges_has_two_faces, slided_mesh
