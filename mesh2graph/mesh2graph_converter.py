import pymesh
from util.meshes import preprocess_mesh
from mesh2graph.graph import GraphConstructor
import warnings
from util.augmentation import scale_verts, rotateY3D, rotateX3D, slide_verts
from util.util import is_mesh_file, mkdir
from mesh2graph.mesh2graph_parser import Mesh2GraphParser
import os
import numpy as np



def generate_augmented_graph_from_single_mesh(mesh_file, hybrid_graph_dir, prevent_nonmanifold_edges=True,
                                              num_of_slide_aug_for_each_sample=0, num_of_jitter_aug_for_each_sample=0,
                                              jitter_rotation=False, mean=0, var=0.005, coef=1, num_y_rotation=0,
                                              rotate_x_90_for_each_y=True, slide_vert_percentage=0.2):

    input_mesh = pymesh.load_mesh(mesh_file)

    # Remove duplicated vertices, duplicated faces, degenerate faces
    # and, if required, faces with non-manifold edges.
    input_mesh = preprocess_mesh(input_mesh=input_mesh, prevent_nonmanifold_edges=prevent_nonmanifold_edges)
    mesh_name = mesh_file.split('/')[-1]

    # Generate the hybrid graph for the original mesh before augmentation
    filename_to_save = os.path.join(hybrid_graph_dir, mesh_name.split('.')[0] + '_0_graph.bin')
    graph_constructor = GraphConstructor(input_mesh=input_mesh, mesh_name=mesh_name,
                                         filename_to_save=filename_to_save, graph_label=None)
    hybrid_graph = graph_constructor.create_graphs()

    # Augmentation:  Randomly shifting the vertices to different locations on the mesh sur-face in the
    # close-to-planar surface region
    # Generate the hybrid graph for augmented mesh
    if slide_vert_percentage != 0:
        for i in range(num_of_slide_aug_for_each_sample):
            all_edges_has_two_faces, slided_mesh = slide_verts(input_mesh, slide_vert_percentage, mesh_name)
            if all_edges_has_two_faces:
                filename_to_save = os.path.join(hybrid_graph_dir, mesh_name.split('.')[0] + '_'
                                                + str(i + 1) + '_slided_graph.bin')
                graph_constructor = GraphConstructor(input_mesh=slided_mesh, mesh_name=mesh_name,
                                                     filename_to_save=filename_to_save,
                                                     graph_label=None)
                hybrid_graph = graph_constructor.create_graphs()

    # Augmentation: Adding a varying Gaussian noise to each vertex of the shape
    # Generate the hybrid graph for augmented mesh
    for i in range(num_of_jitter_aug_for_each_sample):
        augmented_mesh = scale_verts(input_mesh, mean, var, coef)
        filename_to_save = os.path.join(hybrid_graph_dir, mesh_name.split('.')[0] + '_'
                                        + str(i + 1) + '_jittered_graph.bin')
        graph_constructor = GraphConstructor(input_mesh=augmented_mesh, mesh_name=mesh_name,
                                             filename_to_save=filename_to_save,
                                             graph_label=None)
        hybrid_graph = graph_constructor.create_graphs()

    # rotate mesh about the y axis and create a a graph for every theta = (2 * np.pi / num_y_rotation) degree
    for theta in np.linspace(0, 2 * np.pi, num_y_rotation)[1:]:
        rotated_mesh_y = rotateY3D(input_mesh, theta)
        y_theta_deg = int(np.rad2deg(theta))
        filename_to_save = os.path.join(hybrid_graph_dir,
                                        mesh_name.split('.')[0] + '_y_rotation_' +
                                        str(y_theta_deg) + '_graph.bin')
        graph_constructor = GraphConstructor(input_mesh=rotated_mesh_y, mesh_name=mesh_name,
                                             filename_to_save=filename_to_save,
                                             graph_label=None)
        hybrid_graph = graph_constructor.create_graphs()
        # slide vertices for each rotated mesh for more augmentation
        if slide_vert_percentage != 0:
            all_edges_has_two_faces, slided_mesh = slide_verts(rotated_mesh_y, slide_vert_percentage, mesh_name)
            if all_edges_has_two_faces:
                filename_to_save = os.path.join(hybrid_graph_dir,
                                                mesh_name.split('.')[0] + '_y_rotation_' +
                                                str(y_theta_deg) + '_slided_graph.bin')
                graph_constructor = GraphConstructor(input_mesh=slided_mesh, mesh_name=mesh_name,
                                                     filename_to_save=filename_to_save,
                                                     graph_label=None)
                hybrid_graph = graph_constructor.create_graphs()

        if jitter_rotation:
            rotated_mesh_y_jittered = scale_verts(rotated_mesh_y, mean, var, coef)
            filename_to_save = os.path.join(hybrid_graph_dir,
                                            mesh_name.split('.')[0] + '_y_rotation_' +
                                            str(y_theta_deg) + '_jittered_graph.bin')
            graph_constructor = GraphConstructor(input_mesh=rotated_mesh_y_jittered,
                                                 mesh_name=mesh_name,
                                                 filename_to_save=filename_to_save,
                                                 graph_label=None)
            hybrid_graph = graph_constructor.create_graphs()

        if rotate_x_90_for_each_y:
            theta_x = np.pi / 2
            rotated_mesh_y_x_90 = rotateX3D(rotated_mesh_y, theta_x)
            x_theta_deg = int(np.rad2deg(theta_x))
            x_filename_to_save = os.path.join(hybrid_graph_dir, mesh_name.split('.')[0] +
                                              '_y_rotation_' + str(y_theta_deg) + '_x_rotation_' + str(x_theta_deg) +
                                              '_graph.bin')

            graph_constructor = GraphConstructor(input_mesh=rotated_mesh_y_x_90, mesh_name=mesh_name,
                                                 filename_to_save=x_filename_to_save,
                                                 graph_label=None)
            hybrid_graph = graph_constructor.create_graphs()


if __name__ == '__main__':
    converterOpt = Mesh2GraphParser.parse()
    dataset_dir = converterOpt.dataset
    graphs_dir = converterOpt.destination
    # dataset_dir = '/home/bs/Datasets/classification_datasets/ModelNet/Manifold40_manually_aligned_cleaned_translated_scaled_750F'
    # graphs_dir = '/home/bs/Datasets/classification_datasets/ModelNet/Graphs_Manifold40_750F_Aigmented'
    dir_list = os.listdir(dataset_dir)
    categories = [directory for directory in dir_list if os.path.isdir(os.path.join(dataset_dir, directory))]
    manifold_count = 0
    print("Number of classes : {}".format(len(categories)))
    # categories = ['02958343', '02942699', '03797390', '04401088', '02828884', '03759954']
    print(categories)
    i=0
    split_list = ['train', 'test']
    for split in split_list:
        for catagory in categories:
            i=i+1
            category_dir = os.path.join(dataset_dir, catagory, split)
            print("Class number {}, Converting meshes of category {} to a graph".format(i, catagory))
            graph_category_dir = os.path.join(graphs_dir, catagory, split)
            isExist = os.path.exists(graph_category_dir)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(graph_category_dir)
                print("The new directory is created!")
                for item in os.listdir(category_dir):
                    if item != '.ds_store':
                        simplified_mesh = os.path.join(category_dir, item)
                        if os.path.exists(simplified_mesh):
                            print("generating graph for {}".format(simplified_mesh))
                            generate_augmented_graph_from_single_mesh(simplified_mesh,
                                                                graph_category_dir,
                                                                prevent_nonmanifold_edges=False,
                                                                num_of_slide_aug_for_each_sample=0,
                                                                num_of_jitter_aug_for_each_sample=0,
                                                                jitter_rotation=False, mean=0, var=0.005,
                                                                coef=1, num_y_rotation=0, rotate_x_90_for_each_y=False,
                                                                slide_vert_percentage=0.0)
                        else:
                            print("{} doesn't exist".format(simplified_mesh))
