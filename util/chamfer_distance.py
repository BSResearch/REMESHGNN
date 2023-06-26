from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import chamfer_distance
import math
import torch
import numpy as np
from scipy.spatial.distance import cdist


def chamfer_distance(point_cloud1, point_cloud2):
    """Computes the Chamfer distance between two point clouds."""

    # Compute distances between all points in both point clouds
    distances1 = cdist(point_cloud1, point_cloud2)
    distances2 = cdist(point_cloud2, point_cloud1)

    # Compute minimum distance for each point in each point cloud
    min_distances1 = np.min(distances1, axis=1)
    min_distances2 = np.min(distances2, axis=1)

    # Compute Chamfer distance
    chamfer_dist = np.mean(min_distances1) + np.mean(min_distances2)

    return chamfer_dist



# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
def calculate_chamfer_distance(mesh1, mesh2, ymin_is_0 = False, number_of_points=1024):
    verts1, faces1, _ = load_obj(mesh1)
    mesh1 = Meshes(verts=[verts1], faces=[faces1.verts_idx])

    verts2, faces2, _ = load_obj(mesh2)
    mesh2 = Meshes(verts=[verts2], faces=[faces2.verts_idx])

    if ymin_is_0:
        verts1[:, 1] = verts1[:, 1] - torch.min(verts1[:, 1])
        verts2[:, 1] = verts2[:, 1] - torch.min(verts2[:, 1])
        mesh1 = Meshes(verts=[verts1], faces=[faces1.verts_idx])
        mesh2 = Meshes(verts=[verts2], faces=[faces2.verts_idx])

    # Differentiably sample K points from the surface of each mesh and then compute the loss.
    samples1, normal1  = sample_points_from_meshes(mesh1, number_of_points, return_normals=True)
    samples2, normal2 = sample_points_from_meshes(mesh2, number_of_points, return_normals=True)
    loss_chamfer, loss_normal = chamfer_distance(x=samples1, y=samples2, x_lengths=torch.tensor([number_of_points]),
                                                 y_lengths=torch.tensor([number_of_points]), x_normals=normal1,
                                                 y_normals=normal2)
    return loss_chamfer, loss_normal

def select_chamfer_distance(mesh1, mesh2):
    cd_orig, nd_orig = calculate_chamfer_distance(mesh1, mesh2)
    cd_shifted, nd_shifted = calculate_chamfer_distance(mesh1, mesh2, ymin_is_0=True)

    if ((cd_shifted < cd_orig) & (nd_shifted <nd_orig)):
        cd = cd_shifted
        nd = nd_shifted
        shifted = True
    else:
        cd = cd_orig
        nd = nd_orig
        shifted = False
    return cd, nd, shifted

def calculate_bounded_chamfer_distance(mesh1, mesh2, number_of_points=1024):
    verts, faces, _ = load_obj(mesh1)
    mesh1 = Meshes(verts=[verts], faces=[faces.verts_idx])

    verts, faces, _ = load_obj(mesh2)
    mesh2 = Meshes(verts=[verts], faces=[faces.verts_idx])

    # Differentiably sample 2048 points from the surface of each mesh and then compute the loss.
    samples1, normal1 = sample_points_from_meshes(mesh1, number_of_points, return_normals=True)
    samples2, normal2 = sample_points_from_meshes(mesh2, number_of_points, return_normals=True)
    samples1_nn = knn_points(samples1, samples2, K=1)
    samples2_nn = knn_points(samples2, samples1, K=1)
    samples1_dist = 1 - pow(math.e, -1 * (samples1_nn.dists))
    samples2_dist = 1 - pow(math.e, -1 * (samples2_nn.dists))
    cd = (1/samples1.shape[1]) * samples1_dist.sum(1) + (1/samples2.shape[1]) * samples2_dist.sum(1)
    return cd
