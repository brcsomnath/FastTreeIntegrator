
import os
import numpy as np
import os
import copy
import trimesh
import time
from ega.util.mesh_utils import calculate_interpolation_metrics
from numpy import linalg as LA
import sympy as sp 
from typing import List
import random 

from gfi import *
from bfi import *

from ega.util.mesh_utils import  trimesh_to_adjacency_matrices

taylor_expansion_coefficients = lambda k, sigma: [(-1)**i * (sigma)**i / sp.factorial(i) for i in range(k+1)]



class Interpolator():
    """
    given an integrator (for example, brute force), as well as the position of points to be interpolated, 
    this function will do interpolation for these points based on other known points on the mesh graph data 
    """    
    def __init__(self, integrator, vertices_known: List[int], vertices_interpolate: List[int]):
        """ 
        integrator: the integrator to be used. For example: brute_force integrator 
        vertices_known: a list of integers representing the vertices known on the mesh graph, 
                        which are used to predict the fields of the vertices to be interpolated 
        vertices_interpolate: a list of intergers representing the vertices with unknown fields to be interpolated 
        """
        self.integrator = integrator 
        self.vertices_known = vertices_known
        self.vertices_interpolate = vertices_interpolate        
        
    def interpolate(self, field: np.ndarray) -> np.ndarray:
        """ 
        this function predicts the fields for the vertices to be interpolated from existing vertices on the mesh graph data 
        
        inputs: field is an numpy ndarray (for example, a matrix with size N by d representing node features). 
                It can also be a numpy nparray with more than dimension of 2 as long as we can use it as input to integrate_graph_field function 
                    in the integrator.
        """
        # since fields for vertices to be interpolated are unknown, we initialize them as zeros 
        field[self.vertices_interpolate] = 0 
        interpolated_fields = self.integrator.integrate(field)[self.vertices_interpolate]
        return interpolated_fields


def generate_weights_from_adjacency_list(adjacency_lists: List[List[int]], positions: np.ndarray = None,
                                         unweighted = False) -> List[List[int]]:
    """
    given an adjacency list, this function will return the corresponding unweighted list
    (every element equal to 1 in this unweighted list)
    """
    weight_lists = []
    for i, list_i in enumerate(adjacency_lists):
        current_list = []
        for j in list_i:
            if positions is None:
                current_list.append(1)
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
                current_list.append(dist)
        if unweighted:
            weight_lists.append(len(current_list) * [1])
        else:
            weight_lists.append(current_list)

    return weight_lists


def reset_mask_ratio(mesh_data, mask_ratio, unweighted = False):

    adjacency_list = trimesh_to_adjacency_matrices(mesh_data)
    weight_list = generate_weights_from_adjacency_list(adjacency_list, mesh_data.vertices, unweighted = unweighted)

    vertices =  list(np.arange(mesh_data.vertices.shape[0]))  #mesh_data.vertices # mesh_data['vertices']
    all_fields = mesh_data.vertex_normals # the first three columns represent velocities

    world_pos = mesh_data.vertices # mesh_data['world_pos']
    n_vertices = len(vertices)

    # divide vertices into known vertices and vertices to be interpolated
    random.seed(0)
    vertices_interpolate = random.sample(vertices, int(mask_ratio * n_vertices))
    vertices_known = list(set(vertices) - set(vertices_interpolate))
    true_fields = all_fields[vertices_interpolate]
    vertices_interpolate_pos = world_pos[vertices_interpolate]
    n_vertices_interpolate = len(vertices_interpolate)

    # field_known represents the existed fields on the manifold
    # our goal is to predict the fields that is unknown (the points to be interpolated)
    field_known = copy.deepcopy(all_fields)
    field_known[vertices_interpolate] = 0
    ones_known = np.ones(shape = (len(field_known), 1))
    ones_known[vertices_interpolate] = 0

    return adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
        vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
        field_known, ones_known


def main(config):

    meshgraph_file_ids = config['meshgraph_file_ids']
    log_file = config['log_file']
    method_list = config['method_list']
    meshgraph_path = config['meshgraph_path']
    mask_ratio = config['mask_ratio']
    result_file = config['result_file']

    result_list = []

    for mesh_id in meshgraph_file_ids:

        with open(log_file, 'a') as f:
            f.write("\n\n\n")
            f.write("### mesh_id = " + str(mesh_id))

        meshgraph_file = os.path.join(meshgraph_path, str(mesh_id) + '.stl' )

        mesh_data = trimesh.load(meshgraph_file, include_normals=True)
        mesh_data.vertices = mesh_data.vertices - mesh_data.vertices.mean(axis = 0)
        mesh_data.vertices = mesh_data.vertices / np.linalg.norm(mesh_data.vertices, axis = 1).max()

        adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
            vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
            field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio)

        with open(log_file, 'a') as f:
            f.write(", number of nodes = {} ###".format((n_vertices)))

        new_format = [[[adjacency_list[i][j], weight_list[i][j]] for j in range(len(adjacency_list[i]))] for i in range(len(adjacency_list))]
    
    
        ###########################################################################
        ##########################  BruteForceGraphIntegrator  #######################
        ###########################################################################

        if 'BruteForceGraphIntegrator' in method_list:
            best_cosine_similarity = -1
            best_sigma = None
            best_taylor_order = None
            best_preprosess_time = None
            best_interpolation_time = None

            frobenius_norm_list = []
            cosine_similarity_list = []

            bf_sigma_list = config['brute_force']['bf_sigma_list']
            bf_taylor_order_list = config['brute_force']['bf_taylor_order_list']

            for taylor_order in bf_taylor_order_list:
                for sigma in bf_sigma_list:
            
                    if taylor_order == 1:
                        #f_func = lambda x: 1 - sigma * x
                        f_func = lambda x: 1 / (1 + sigma * x)
                        
                    elif taylor_order == 2:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2
                    elif taylor_order == 3:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6
                    elif taylor_order == 4:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6 + sigma**4 * x**4 / 24                
                    elif taylor_order == 5:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6 + sigma**4 * x**4 / 24 - sigma**5 * x**5 / 120                 
                    elif taylor_order == 6:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6 + sigma**4 * x**4 / 24 - sigma**5 * x**5 / 120 + sigma**6 * x**6 / 720                
                    
                    
                    ## pre-processing time
                    start = time.time()
                    tree = kruskal_algo(new_format)
                    tbgi = BruteForceGraphIntegrator(f_func, True, tree)
                    interpolator_bf = Interpolator(tbgi, vertices_known, vertices_interpolate)
                    
                    preprocess_time = time.time() - start
                    with open(log_file, 'a') as f:
                        f.write("\tPreprocessing takes: {} seconds".format(time.time() - start))

                    ## interpolation time
                    start = time.time()
                    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                    interpolator_bf.interpolate(copy.deepcopy(ones_known))
                    interpolated_fields_bf = interpolated_fields_bf / LA.norm(interpolated_fields_bf, axis = 1, keepdims=True)
                    interpolation_time = time.time() - start

                    frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf, verbose=False)
                    with open(log_file, 'a') as f:
                        f.write("\n")
                        f.write("\tInterpolation takes: {} seconds".format(time.time() - start))
                        f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                        f.write("\tfrobenius_norm is: {}".format(frobenius_norm))

                    if cosine_similarity > best_cosine_similarity:
                        best_cosine_similarity = cosine_similarity
                        best_sigma = sigma 
                        best_taylor_order = taylor_order
                        best_preprocess_time = preprocess_time
                        best_interpolation_time = interpolation_time

                    frobenius_norm_list.append(frobenius_norm)
                    cosine_similarity_list.append(cosine_similarity)


                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("Taylor order is: {}, Best sigma is: {}, best cosine similarity is: {}".format(taylor_order, best_sigma, best_cosine_similarity))

                result_list.append([mesh_id, taylor_order, best_cosine_similarity, best_preprocess_time, best_interpolation_time])
                np.savetxt(result_file, result_list, delimiter=",")


        ###########################################################################
        ##########################  TreeBasedGraphIntegrator  #######################
        ###########################################################################

        if 'TreeBasedGraphIntegrator' in method_list:
            
            best_cosine_similarity = -1
            best_sigma = None
            best_taylor_order = None
            best_preprosess_time = None
            best_interpolation_time = None

            frobenius_norm_list = []
            cosine_similarity_list = []

            ours_sigma_list = config['tree_based']['tree_sigma_list']
            ours_taylor_order_list = config['tree_based']['tree_taylor_order_list']

            for taylor_order in ours_taylor_order_list:
                for sigma in ours_sigma_list:
            
                    if taylor_order == 1:
                        #f_func = lambda x: 1 - sigma * x
                        f_func = lambda x: 1 / (1 + sigma * x)
                    elif taylor_order == 2:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2
                    elif taylor_order == 3:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6
                    elif taylor_order == 4:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6 + sigma**4 * x**4 / 24                
                    elif taylor_order == 5:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6 + sigma**4 * x**4 / 24 - sigma**5 * x**5 / 120                 
                    elif taylor_order == 6:
                        f_func = lambda x: 1 - sigma * x + sigma**2 * x**2 / 2 - sigma**3 * x**3 / 6 + sigma**4 * x**4 / 24 - sigma**5 * x**5 / 120 + sigma**6 * x**6 / 720                
                    
                    
                    ## pre-processing time
                    start = time.time()
                    tree = kruskal_algo(new_format)
                    tbgi = TreeBasedGraphIntegrator(f_func, True, tree, MinimumSpanningTreeConstructor(), threshold=int(0.25*n_vertices))

                    interpolator_bf = Interpolator(tbgi, vertices_known, vertices_interpolate)

                    preprocess_time = time.time() - start
                    with open(log_file, 'a') as f:
                        f.write("\tPreprocessing takes: {} seconds".format(time.time() - start))

                    ## interpolation time
                    start = time.time()
                    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                    interpolator_bf.interpolate(copy.deepcopy(ones_known))
                    interpolated_fields_bf = interpolated_fields_bf / LA.norm(interpolated_fields_bf, axis = 1, keepdims=True)
                    interpolation_time = time.time() - start

                    frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf, verbose=False)
                    with open(log_file, 'a') as f:
                        f.write("\n")
                        f.write("\tInterpolation takes: {} seconds".format(time.time() - start))
                        f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                        f.write("\tfrobenius_norm is: {}".format(frobenius_norm))

                    if cosine_similarity > best_cosine_similarity:
                        best_cosine_similarity = cosine_similarity
                        best_sigma = sigma 
                        best_taylor_order = taylor_order
                        best_preprocess_time = preprocess_time
                        best_interpolation_time = interpolation_time

                    frobenius_norm_list.append(frobenius_norm)
                    cosine_similarity_list.append(cosine_similarity)


                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("Taylor order is: {}, Best sigma is: {}, best cosine similarity is: {}".format(taylor_order, best_sigma, best_cosine_similarity))

                print(f"Mesh Id: {mesh_id}\n   Taylor order: {taylor_order} Best sigma: {best_sigma}  ==>  Preprocess Time: {best_preprocess_time}, Interpolation Time: {best_interpolation_time}, Cosine_sim: {np.round(best_cosine_similarity, 4)}")

                result_list.append([mesh_id, taylor_order, best_cosine_similarity, best_preprocess_time, best_interpolation_time])
                np.savetxt(result_file, result_list, delimiter=",")




#%%
    
if __name__=="__main__":
    

    config = {
                 "tree_based":  {'tree_sigma_list': [100, 100000], 'tree_taylor_order_list': [1]},             
                'log_file': './log.txt',
                'mask_ratio': 0.8,
                'meshgraph_file_ids': [60246, 85580, 40179, 964933, 1624039, 91657, 79183, 82407, 40172, 65414, 90431, 74449, 73464, 230349, 40171, 61193, 77938, 375276, 39463, 110793, 368622, 37326, 42435, 1514901, 65282, 116878, 550964, 409624, 101902, 73410, 87602, 255172, 98480, 57140, 285606, 96123, 203289, 87601, 409629, 37384, 57084],
                'meshgraph_path': "", # set this as the folder where thingi10k meshes are stored
                'method_list': ['TreeBasedGraphIntegrator'],
                'result_file': f'./result.csv',
            }
    
    
    main(config)
    
        

