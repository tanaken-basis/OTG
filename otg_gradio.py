##### OTG (Optimal Transport Grouping)
## tanaken ( Kentaro TANAKA, 2024.2- )

#### WebUI with Gradio ####

############################################################
#### Required libraries ####
# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install umap-learn
# pip install gradio

############################################################
#### Import ####
import copy
import math
import time
import random
import datetime
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import cm as cm
import gradio as gr
is_umap_loaded = True
try:
    import umap
    # from numba import jit
except ImportError as e:
    is_umap_loaded = False
    print(f"{e} is not installed.")

############################################################
#### Functions ####

## Functions for manipulating tensors
def get_N(N_size):
    if len(N_size) < 1:
        raise ValueError("Error: N_size is invalid.")
    N_rank = len(N_size)
    N_accum = np.ones(N_rank, dtype=int) # n2*n3*...*nN, n3*n4*...*nN, ... , nN, 1
    for i in range(N_rank):
        if i==0:
            N_accum[N_rank-i-1] = 1
        else:
            N_accum[N_rank-i-1] = N_accum[N_rank-i]*N_size[N_rank-i]
    N_size_prod = N_size[0]*N_accum[0]
    return (N_rank, N_accum, N_size_prod)

def get_tensor_flattened_index_from_multi_index(multi_index, N_rank, N_accum):
    flattened_index = 0
    for i in range(N_rank):
            flattened_index = flattened_index + N_accum[i]*multi_index[i]
    flattened_index = int(flattened_index)
    return flattened_index

def get_tensor_multi_index_from_flattened_index(flattened_index, N_rank, N_accum):
    multi_index = []
    remainder = flattened_index
    for i in range(N_rank):
        quotient, remainder = divmod(remainder, N_accum[i])
        multi_index.append(quotient)
    multi_index = tuple(multi_index)
    return multi_index

def get_tensor_value_from_multi_index(target_tensor, multi_index, N_rank, N_accum):
    flattened_index = get_tensor_flattened_index_from_multi_index(multi_index, N_rank, N_accum)
    return target_tensor[flattened_index]

def get_tensor_flattened_index_list_from_value(target_tensor, value, tensor_tolerance=None):
    if (tensor_tolerance is None) or (tensor_tolerance==0):
        return [i for i, element in enumerate(target_tensor) if element==value]
    else:
        return [i for i, element in enumerate(target_tensor) if abs(element-value)<=tensor_tolerance]

def get_tensor_multi_index_list_from_value(target_tensor, value, N_rank, N_accum, tensor_tolerance=None):
    multi_index_list = []
    flattened_index_list = get_tensor_flattened_index_list_from_value(target_tensor, value, tensor_tolerance)
    for flattened_index in flattened_index_list:
        multi_index_list.append(get_tensor_multi_index_from_flattened_index(flattened_index, N_rank, N_accum))
    return multi_index_list

## Function to generate marginal mass vectors
def calc_marginal_mass_vectors(N_rank, N_size):
    marginal_mass_vectors = []
    for i in range(N_rank):
        marginal_mass_vectors.append(np.ones(N_size[i])/N_size[i])
    return marginal_mass_vectors

## Function to generate grouping randomly (rand=True) or in the same order as the data_order_list (rand=False)
def gen_grouping_indexes_list(N_size, rand=True, data_order_list=None):
    if data_order_list is None:
        data_order_list = list(range(sum(N_size)))
    if rand:
        data_order_list = random.sample(data_order_list, len(data_order_list))
    grouping_indexes_list = [] # Double listing for grouping
    range_from = 0
    range_to = 0
    for size in N_size:
        range_to = range_from + size
        grouping_indexes_list.append(data_order_list[range_from:range_to])
        range_from = range_to
    return grouping_indexes_list

## Functions to calculate costs of optrimal transport, etc.
# @jit
def calc_multi_ot(marginal_mass_vectors, cost_tensor, normalized_cost_tensor,
                  N_size, N_rank, N_accum, N_size_prod,
                  numerical_precision=2e-8, ot_speed=0.02, ot_stopping_rule=0.02, ot_loop_max=200): ## ot_stopping_rule: Criteria to stop updating "u". If the relative error of "u" is smaller than the stop criterion, it is terminated.
    ## Optrimal transport
    K_tensor = np.exp(- normalized_cost_tensor / ot_speed) # Gibbs kernel
    u_vec_list = []
    for i in range(N_rank):
        u_vec_list.append(np.ones(N_size[i]))
    for loop in range(ot_loop_max):
        u_diff = 0 # Variable to measure whether to exit the loop
        for i in range(N_rank):
            for j in range(N_size[i]):
                temp_u_value = 0
                temp_K_value = 1
                temp_u_prod_value = 1
                N_sizeub_s = list(copy.copy(N_size))
                N_sizeub_s.pop(i)
                for m_sub_index in np.ndindex(tuple(N_sizeub_s)):
                    temp_m_index = list(copy.copy(m_sub_index))
                    temp_m_index.insert(i, j)
                    temp_K_value = get_tensor_value_from_multi_index(K_tensor, temp_m_index, N_rank, N_accum)
                    temp_u_prod_value = 1
                    for k in range(N_rank):
                        if k != i:
                            temp_u_prod_value = temp_u_prod_value * u_vec_list[k][temp_m_index[k]]
                    temp_u_value = temp_u_value + temp_K_value * temp_u_prod_value
                temp_u_value = (marginal_mass_vectors[i][j]) / (temp_u_value)
                u_diff = max(u_diff, abs((u_vec_list[i][j]-temp_u_value)/(temp_u_value+numerical_precision))) 
                u_vec_list[i][j] = temp_u_value
        if abs(u_diff) < ot_stopping_rule:
            break
    f_vec_list = []
    for i in range(N_rank):
        temp_f_vec = ot_speed * np.log(u_vec_list[i] + numerical_precision)
        f_vec_list.append(temp_f_vec)
    P_tensor = np.zeros(N_size_prod)
    weighted_cost_tensor = np.zeros(N_size_prod)
    objective_function_value = 0
    for m_index in np.ndindex(tuple(N_size)):
        temp_cost_value = get_tensor_value_from_multi_index(cost_tensor, m_index, N_rank, N_accum)
        temp_P_value = get_tensor_value_from_multi_index(K_tensor, m_index, N_rank, N_accum)
        for k in range(N_rank):
            temp_P_value = temp_P_value * u_vec_list[k][m_index[k]]
        P_tensor[get_tensor_flattened_index_from_multi_index(m_index, N_rank, N_accum)] = temp_P_value
        weighted_cost_tensor[get_tensor_flattened_index_from_multi_index(m_index, N_rank, N_accum)] = temp_P_value*temp_cost_value
        objective_function_value = objective_function_value + weighted_cost_tensor[get_tensor_flattened_index_from_multi_index(m_index, N_rank, N_accum)]
    return (objective_function_value, P_tensor, weighted_cost_tensor, u_vec_list, f_vec_list)

def calc_intergroup_cost_tensor(grouping_indexes_list, data_points_nparray, marginal_mass_vectors,
                                N_size, N_rank, N_accum, N_size_prod,
                                numerical_precision=2e-8):
    cost_tensor = np.zeros(N_size_prod)
    for m_index in np.ndindex(N_size):
        temp_data_points_nparray = []
        temp_cost_value = 0
        ## Cost : Sum of distances (not squared) between each point and the barycenter
        for group in range(N_rank):
            temp_data_points_nparray.append(data_points_nparray[grouping_indexes_list[group][m_index[group]]])
        temp_barycenter = np.mean(temp_data_points_nparray, axis=0)
        for group in range(N_rank):
            temp_cost_value_bt2 = np.linalg.norm(temp_data_points_nparray[group] - temp_barycenter) ## Cost between two points
            temp_cost_value = temp_cost_value + temp_cost_value_bt2
        temp_index = get_tensor_flattened_index_from_multi_index(m_index, N_rank, N_accum)
        cost_tensor[temp_index] = temp_cost_value
    normalized_cost_tensor = copy.deepcopy(cost_tensor)
    max_cost_value = max(cost_tensor)
    if max_cost_value > numerical_precision:
        normalized_cost_tensor = normalized_cost_tensor/max_cost_value
    return (cost_tensor, normalized_cost_tensor)

def calc_intergroup_cost_value(grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
                               N_size, N_rank, N_accum, N_size_prod,
                               numerical_precision=2e-8, ot_speed=0.02, ot_stopping_rule=0.02, ot_loop_max=200):
    (intergroup_cost_tensor, normalized_intergroup_cost_tensor) = calc_intergroup_cost_tensor(
        grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
        N_size, N_rank, N_accum, N_size_prod,
        numerical_precision)
    (intergroup_cost_value, intergroup_P_tensor, intergroup_weighted_cost_tensor, 
     intergroup_u_vec_list, intergroup_f_vec_list) = calc_multi_ot(
        marginal_mass_vectors, intergroup_cost_tensor, normalized_intergroup_cost_tensor, N_size, N_rank, N_accum, N_size_prod,
        numerical_precision, ot_speed, ot_stopping_rule, ot_loop_max)
    return (intergroup_cost_value, intergroup_P_tensor, intergroup_weighted_cost_tensor, 
            intergroup_u_vec_list, intergroup_f_vec_list, intergroup_cost_tensor)

def calc_intragroup_cost_nparray_list(grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
                                      N_size, N_rank, N_accum, N_size_prod):
    cost_nparray_list = []
    barycenter_nparray_list = []
    for group, size in enumerate(N_size):
        temp_cost_nparray = np.zeros(size)
        for element in range(size):
            temp_data_points_nparray = []
            ## Cost : Sum of distances (not squared) between each point and the barycenter
            for element in range(N_size[group]): ## barycenter
                temp_data_points_nparray.append(data_points_nparray[grouping_indexes_list[group][element]])
            temp_barycenter_nparray = np.mean(temp_data_points_nparray, axis=0)
            for element in range(N_size[group]): ## Cost between one mass point and barycenter
                temp_cost_value_bt2 = np.linalg.norm(temp_data_points_nparray[element] - temp_barycenter_nparray) ## Cost between two points
                temp_cost_nparray[element] = temp_cost_value_bt2
        cost_nparray_list.append(temp_cost_nparray)
        barycenter_nparray_list.append(temp_barycenter_nparray)
    return (cost_nparray_list, barycenter_nparray_list)

def calc_intragroup_cost_value(grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
                               N_size, N_rank, N_accum, N_size_prod):
    intragroup_cost_value = 0
    intragroup_average_cost_list = []
    (intragroup_cost_nparray_list, intragroup_barycenter_nparray_list) = calc_intragroup_cost_nparray_list(
        grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
        N_size, N_rank, N_accum, N_size_prod
        )
    for group in range(N_rank):
        intragroup_average_cost = np.mean(intragroup_cost_nparray_list[group])
        intragroup_average_cost_list.append(intragroup_average_cost)
        intragroup_cost_value = intragroup_cost_value + intragroup_average_cost
    intragroup_cost_value = intragroup_cost_value/N_rank
    return (intragroup_cost_value, intragroup_cost_nparray_list, intragroup_average_cost_list, intragroup_barycenter_nparray_list)

def calc_aggregate_statistical_cost_list(intragroup_barycenter_nparray_list, intragroup_average_cost_list,
                                         N_size, N_rank, N_accum, N_size_prod):
    center_of_intragroup_barycenter_nparray_list =  np.mean(intragroup_barycenter_nparray_list, axis=0)
    center_of_intragroup_average_cost = np.mean(intragroup_average_cost_list, axis=0)
    mean_cost_value = 0
    variance_cost_value = 0
    for group in range(N_rank):
        mean_cost_value = mean_cost_value + np.linalg.norm(intragroup_barycenter_nparray_list[group] - center_of_intragroup_barycenter_nparray_list)
        variance_cost_value = variance_cost_value + abs(intragroup_average_cost_list[group] - center_of_intragroup_average_cost)
    mean_cost_value = mean_cost_value/N_rank
    variance_cost_value = variance_cost_value/N_rank   
    return (mean_cost_value, variance_cost_value)

def calc_adjusted_cost_value(grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
                             N_size, N_rank, N_accum, N_size_prod, 
                             mean_penalty_weight=0.2, variance_penalty_weight=0.8, 
                             numerical_precision=2e-8, ot_speed=0.02, ot_stopping_rule=0.02, ot_loop_max=200):
    ## intergroup_cost_value
    (intergroup_cost_value, intergroup_P_tensor, intergroup_weighted_cost_tensor, 
    intergroup_u_vec_list, intergroup_f_vec_list,
    intergroup_cost_tensor) = calc_intergroup_cost_value(
        grouping_indexes_list, data_points_nparray, marginal_mass_vectors,
        N_size, N_rank, N_accum, N_size_prod,
        numerical_precision, ot_speed, ot_stopping_rule, ot_loop_max
        )
    ## intragroup_cost_value
    (intragroup_cost_value, intragroup_cost_nparray_list, intragroup_average_cost_list,
     intragroup_barycenter_nparray_list) = calc_intragroup_cost_value(
        grouping_indexes_list, data_points_nparray, marginal_mass_vectors,
        N_size, N_rank, N_accum, N_size_prod
        )
    ## aggregate_statistical_cost_value
    (mean_cost_value, variance_cost_value) = calc_aggregate_statistical_cost_list(
        intragroup_barycenter_nparray_list, intragroup_average_cost_list,
        N_size, N_rank, N_accum, N_size_prod
        )
    ## adjusted_cost_value = (intergroup_cost_value + mean_cost_value + variance_cost_value) / (intragroup_cost_value)
    adjusted_cost_value = 0
    if abs(intragroup_cost_value) < numerical_precision:
        adjusted_cost_value = np.inf
    else:
        adjusted_cost_value = (intergroup_cost_value + mean_penalty_weight*mean_cost_value + variance_penalty_weight*variance_cost_value)/(intragroup_cost_value)
    ## return
    return (adjusted_cost_value, mean_cost_value, variance_cost_value,
            intragroup_cost_value, intragroup_cost_nparray_list, intragroup_barycenter_nparray_list, 
            intergroup_cost_value, intergroup_P_tensor, intergroup_weighted_cost_tensor, 
            intergroup_u_vec_list, intergroup_f_vec_list, intergroup_cost_tensor)

## Functions for drawing graphs
def get_rank_vec(v):
    n = len(v)
    rank = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if v[i]>v[j]:
                rank[j] = rank[j] + 1
            else:
                rank[i] = rank[i] + 1
    return rank

def get_points_list_in_non_intersecting_order(x, y):
    n = len(x)
    qcos_vec = [0]*n
    reference_index = y.index(min(y))
    qcos_vec[reference_index] = 2
    indices_rem = (list(range(n)))
    indices_rem.pop(reference_index)
    range_x = max(x)-min(x)
    for i in indices_rem:
        dx = (x[i]-x[reference_index])
        dy = (y[i]-y[reference_index])
        dr = math.sqrt(dx*dx + dy*dy)
        if dr == 0:
            qcos_vec[i] = 2
        elif dx == dr:
            qcos_vec[i] = 2 - dx/range_x
        elif dx == -dr:
            qcos_vec[i] = -2 - dx/range_x
        else:
            qcos_vec[i] = dx/dr
    rank_cos = get_rank_vec(qcos_vec)
    points = [[]]*n
    for i in range(n):
        points[rank_cos[i]] = [x[i], y[i]]
    return points

def show_P_tensor(P_tensor, N_size, N_rank, N_accum, f_size=(6,4), numerical_precision=1e-8, f_title=""):
    ## Draw the graph of the tensor of the solution of the grouping
    ## The horizontal axis is the groups (1～N_rank) and the vertical axis is the number of elements belonging to each group (N_size).
    ## A single line corresponds to one element of the tensor. The higher the value, the thicker the line.
    ## ------------------------------
    ## Visualization of P_tensor
    x = list(range(N_rank))
    x = [element+1 for element in x]
    fig = plt.figure(figsize = (f_size[0], f_size[1]), facecolor="mistyrose")
    ax = fig.add_subplot(111)
    ax.set_title(f_title)
    ax.set_xlim((0, N_rank+1))
    ax.set_ylim((0, max(N_size)+1))
    P_max = max(P_tensor)
    for m_index in np.ndindex(tuple(N_size)):
        temp_y = list(m_index)
        temp_y = [element+1 for element in temp_y]
        temp_P_ratio = P_tensor[get_tensor_flattened_index_from_multi_index(m_index, N_rank, N_accum)] / (P_max + numerical_precision)
        lwd = 10*math.sqrt(temp_P_ratio) # 10 * math.log( math.exp(1) - 1 + temp_P_ratio　)
        ax.plot(x, temp_y, linewidth=lwd, alpha=0.5)

def gen_2d_data(is_umap_loaded, data_points_nparray):
    if len(data_points_nparray[0,:]) > 2:
        if is_umap_loaded:
            ## Umap
            print("Umapping...", flush="True")
            mapper = umap.UMAP(random_state=0)
            embedding = mapper.fit_transform(data_points_nparray)
            return (embedding[:,0], embedding[:,1])
        else:
            print("For two-dimensional visualization, use only the first and second variables.")
            return (data_points_nparray[:,0], data_points_nparray[:,1])
    elif len(data_points_nparray[0,:]) == 2:
        return (data_points_nparray[:,0], data_points_nparray[:,1])
    elif len(data_points_nparray[0,:]) == 1:
        return (data_points_nparray[:,0], np.zeros(len(data_points_nparray[:,0])))
    else:
        return ([0], [0])

def show_2d_data(is_umap_loaded, grouping_indexes_list, data_points_nparray,
                 viz2d_x_value = None, viz2d_y_value = None, line_width = 1, f_size=(8,6,4), f_title=""):
    ## Visualization of two-dimensional data
    ## Each group is arranged in order of starting point (tau) to ending point (nu).
    len_data_points_nparray = len(data_points_nparray)
    if len_data_points_nparray > 20:
        line_width = 0
    if grouping_indexes_list is None:
        grouping_indexes_list = [list(range(len_data_points_nparray))]
    if (viz2d_x_value is None) or (viz2d_y_value is None):
        viz2d_x_value, viz2d_y_value = gen_2d_data(is_umap_loaded, data_points_nparray)
    x_min = min(viz2d_x_value)
    x_max = max(viz2d_x_value)
    x_range = x_max - x_min
    y_min = 0
    y_max = 0
    y_range = 0
    if len(data_points_nparray[0,:]) > 1:
        viz2d_y_value = viz2d_y_value
        y_min = min(viz2d_y_value)
        y_max = max(viz2d_y_value)
        y_range = y_max - y_min
    figsize_x = f_size[0]
    figsize_y = f_size[1]
    if max(x_range, y_range) <= 0:
        figsize_x = f_size[0]
        figsize_y = f_size[1]
    elif x_range > y_range:
        figsize_x = f_size[0]
        figsize_y = max(f_size[2] ,min(f_size[1], f_size[1]*(y_range/x_range)))
    else:
        figsize_y = f_size[1]
        figsize_x = max(f_size[2] ,min(f_size[1], f_size[1]*(x_range/y_range)))
    fig = plt.figure(figsize = (figsize_x, figsize_y), facecolor="mistyrose")
    ax = fig.add_subplot(111)
    ax.set_xlim((x_min-2, x_max+2))
    ax.set_ylim((y_min-2, y_max+2))
    colors = cm.tab10 # cm.tab20
    len_colors = 10
    markers = ["o", "^", "s", "p", "D", "H", "*", "v", "<", ">",  
                "+", "x", ".", ",", "d", "h", "1", "2", "3", "4", "8", "|", "_"]
    for group, index_list in enumerate(grouping_indexes_list):
        if len(index_list)>0:
            x = []
            y = []
            x_start = [] # Starting point: tau
            y_start = [] # Starting point: tau
            x_end = [] # Ending point: nu
            y_end = [] # Ending point: nu
            p_color = colors(int(group)%len_colors)
            p_marker = markers[int(group)%len(markers)]
            for j, element in enumerate(index_list):
                x.append(viz2d_x_value[element])
                y.append(viz2d_y_value[element])
                if j==0:
                    x_start = [viz2d_x_value[element]]
                    y_start = [viz2d_y_value[element]]
                elif j==(len(index_list)-1):
                    x_end = [viz2d_x_value[element]]
                    y_end = [viz2d_y_value[element]]
            ax.plot(x, y, alpha=0.5, color=p_color, marker=p_marker, markersize=12, linewidth=line_width)
            ax.set_title(f_title)
            if line_width > 0:
                ## Starting point: tau
                ax.plot(x_start, y_start, alpha=0.5, marker="$\\tau$", markersize=6, color="black")
                ## Ending point: nu
                ax.plot(x_end, y_end, alpha=0.5, marker="$\\nu$", markersize=6, color="black")
    return (fig, ax, viz2d_x_value, viz2d_y_value)

def show_2d_data_with_patches(is_umap_loaded, grouping_indexes_list, data_points_nparray, 
                              N_size, N_rank, N_accum, N_size_prod,
                              viz2d_x_value = None, viz2d_y_value = None, patch_weight = None, line_width = 1, f_size=(8,6,4), f_title=""):
    (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, grouping_indexes_list, data_points_nparray,
                                               viz2d_x_value, viz2d_y_value,
                                               line_width = 1, f_size=(8,6,4), f_title="")
    if patch_weight is not None:
        patch_weight_max = max(patch_weight)
        if patch_weight_max > 0:
            if (N_rank is None) or (N_accum is None) or (N_size_prod is None):
                (N_rank, N_accum, N_size_prod) = get_N(N_size)
            for m_index in np.ndindex(N_size):
                w = get_tensor_value_from_multi_index(patch_weight, m_index, N_rank, N_accum)
                alp = w / patch_weight_max
                alp = 0.5 * alp / N_rank
                if alp > 0.001:
                    x_vec = []
                    y_vec = []
                    for group in range(N_rank):
                        index_value = grouping_indexes_list[group][m_index[group]]
                        x_vec.append(viz2d_x_value[index_value])
                        y_vec.append(viz2d_y_value[index_value])
                    if N_rank > 2:
                        points = get_points_list_in_non_intersecting_order(x_vec, y_vec)
                        patch = patches.Polygon(xy=points, closed=True, alpha=alp, color='black')
                        ax.add_patch(patch)
                    elif N_rank == 2:
                        ax.plot(x_vec, y_vec, alpha=alp, color='black',
                                marker=None, linestyle='solid', linewidth=2)
    return (fig, ax, viz2d_x_value, viz2d_y_value)

def get_argmax_list(target_tensor, fixed_group_list, fixed_element_list, 
                    N_size, N_rank, N_accum):
    N_size_partially_fixed = copy.deepcopy(N_size)
    N_size_partially_fixed = list(N_size_partially_fixed)
    if (len(fixed_group_list)!=0) and (len(fixed_group_list)==len(fixed_element_list)):
        for group in fixed_group_list:
            N_size_partially_fixed[group] = 1
    N_size_partially_fixed = tuple(N_size_partially_fixed)
    temp_max = 0
    argmax_list = []
    for m_index in np.ndindex(N_size_partially_fixed):
        m_index_list = list(m_index)
        if (len(fixed_group_list)!=0) and (len(fixed_group_list)==len(fixed_element_list)):
            for element, group in enumerate(fixed_group_list):
                m_index_list[group] = fixed_element_list[element]
        temp_value = target_tensor[get_tensor_flattened_index_from_multi_index(m_index_list, N_rank, N_accum)]
        if temp_value == temp_max:
            argmax_list.append(m_index_list)
        elif temp_value > temp_max:
            temp_max = temp_value
            argmax_list = [m_index_list]
    return argmax_list

def get_marginal_value(target_tensor, fixed_group_list, fixed_element_list, 
                       N_size, N_rank, N_accum):
    N_size_partially_fixed = copy.deepcopy(N_size)
    N_size_partially_fixed = list(N_size_partially_fixed)
    if (len(fixed_group_list)!=0) and (len(fixed_group_list)==len(fixed_element_list)):
        for group in fixed_group_list:
            N_size_partially_fixed[group] = 1
    N_size_partially_fixed = tuple(N_size_partially_fixed)
    marginal_value = 0
    for m_index in np.ndindex(N_size_partially_fixed):
        m_index_list = list(m_index)
        if (len(fixed_group_list)!=0) and (len(fixed_group_list)==len(fixed_element_list)):
            for element, group in enumerate(fixed_group_list):
                m_index_list[group] = fixed_element_list[element]
        temp_value = target_tensor[get_tensor_flattened_index_from_multi_index(m_index_list, N_rank, N_accum)]
        marginal_value = marginal_value + temp_value
    return marginal_value

## Functions to calculate optrimal grouping
def calc_optimal_grouping(data_points_nparray, N_size,
                           N_rank = None, N_accum = None, N_size_prod = None,
                           mean_penalty_weight = 0.2, variance_penalty_weight = 0.8,
                           numerical_precision = 2e-8,
                           ot_speed = 0.02, ot_stopping_rule = 0.02, ot_loop_max = 200,
                           tensor_tolerance = 2e-8, global_loop_max = 10, local_loop_max = 100,
                           init_grouping_indexes_list = None, init_grouping_rand = True,
                           search_method = "ex", search_stopping_rule_err = 0.02, search_stopping_rule_rep = 20,
                           show_info = False, drawing_graphs = False,
                           info_func = (lambda info_args, txt: print(str(txt))),
                           info_args = None,
                           viz2d_x_value = None, viz2d_y_value = None):
    ## N_rank, N_accum, N_size_prod, marginal_mass_vectors
    if (N_rank is None) or (N_accum is None) or (N_size_prod is None):
        (N_rank, N_accum, N_size_prod) = get_N(N_size)
    marginal_mass_vectors = calc_marginal_mass_vectors(N_rank, N_size)
    ## Initial value settings
    if init_grouping_indexes_list is None:
        init_grouping_indexes_list = gen_grouping_indexes_list(N_size, rand=init_grouping_rand) ## True: Random grouping, False: Grouping in order
    ## Calculation of optimal transportation costs under initial conditions
    (init_adjusted_cost_value, init_mean_cost_value, init_variance_cost_value,
    init_intragroup_cost_value, init_intragroup_cost_nparray_list, init_intragroup_barycenter_nparray_list,
    init_intergroup_cost_value, init_intergroup_P_tensor, init_intergroup_weighted_cost_tensor,
    init_intergroup_u_vec_list, init_intergroup_f_vec_list,
    init_intergroup_cost_tensor) = calc_adjusted_cost_value(
        init_grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
        N_size, N_rank, N_accum, N_size_prod,
        mean_penalty_weight, variance_penalty_weight, 
        numerical_precision, ot_speed, ot_stopping_rule, ot_loop_max
    )
    ## Preparation for recording
    iteration_number_list = [0]
    elapsed_time_list = [0]
    new_adjusted_cost_trends_list = [init_adjusted_cost_value]
    opt_adjusted_cost_trends_list = [init_adjusted_cost_value]
    start_time = time.time()
    ## info
    if show_info:
        info_func(info_args, "---------- init")
        info_func(info_args, "init_grouping_indexes_list: " + str(init_grouping_indexes_list))
        info_func(info_args, "init_adjusted_cost_value: " + str(init_adjusted_cost_value))
        info_func(info_args, "  (init_intergroup_cost_value, init_intragroup_cost_value: " + str(init_intergroup_cost_value) + ", " + str(init_intragroup_cost_value) + ")")
        info_func(info_args, "  (mean_penalty_weight*init_mean_cost_value, variance_penalty_weight*init_variance_cost_value : " 
              + str(mean_penalty_weight*init_mean_cost_value) + ", " + str(variance_penalty_weight*init_variance_cost_value) + ")")
    if drawing_graphs:
        (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data_with_patches(is_umap_loaded, 
                                                                init_grouping_indexes_list, data_points_nparray, 
                                                                N_size, N_rank, N_accum, N_size_prod,
                                                                viz2d_x_value, viz2d_y_value, init_intergroup_P_tensor)
        # (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, init_grouping_indexes_list, data_points_nparray,
        #                             viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(5,4,2), f_title="Initial Value")
        show_P_tensor(init_intergroup_P_tensor, N_size, N_rank, N_accum, f_size=(4,3), f_title="Initial Value")
    ## opt
    opt_grouping_indexes_list = copy.deepcopy(init_grouping_indexes_list)
    opt_adjusted_cost_value = init_adjusted_cost_value
    opt_mean_cost_value = init_mean_cost_value
    opt_variance_cost_value = init_variance_cost_value
    opt_intragroup_cost_value = init_intragroup_cost_value
    opt_intragroup_cost_nparray_list = copy.deepcopy(init_intragroup_cost_nparray_list)
    opt_intragroup_barycenter_nparray_list = copy.deepcopy(init_intragroup_barycenter_nparray_list)
    opt_intergroup_cost_value = init_intergroup_cost_value
    opt_intergroup_P_tensor = copy.deepcopy(init_intergroup_P_tensor)
    opt_intergroup_weighted_cost_tensor = copy.deepcopy(init_intergroup_weighted_cost_tensor)
    opt_intergroup_cost_tensor = copy.deepcopy(init_intergroup_cost_tensor)
    ## new
    new_grouping_indexes_list = copy.deepcopy(init_grouping_indexes_list)
    new_adjusted_cost_value = init_adjusted_cost_value
    new_mean_cost_value = init_mean_cost_value
    new_variance_cost_value = init_variance_cost_value
    new_intragroup_cost_value = init_intragroup_cost_value
    new_intragroup_cost_nparray_list = copy.deepcopy(init_intragroup_cost_nparray_list)
    new_intragroup_barycenter_nparray_list = copy.deepcopy(init_intragroup_barycenter_nparray_list)
    new_intergroup_cost_value = init_intergroup_cost_value
    new_intergroup_P_tensor = copy.deepcopy(init_intergroup_P_tensor)
    new_intergroup_weighted_cost_tensor = copy.deepcopy(init_intergroup_weighted_cost_tensor)
    new_intergroup_cost_tensor = copy.deepcopy(init_intergroup_cost_tensor)
    ## Search for optimal value
    new_grouping_flag = True
    search_stopping_rule_counter = 0
    for loop in range(global_loop_max):
        if show_info:
            info_func(info_args, "---------- loop: " + str(loop+1))
        search_stopping_rule_counter = search_stopping_rule_counter + 1
        if search_method=="rand": ## search_method=="rand"
            new_grouping_indexes_list = gen_grouping_indexes_list(N_size, rand=True) ## True: Random grouping, False: Grouping in order
        else: ## search_method=="ex" or search_method=="hybrid"
            if (search_stopping_rule_counter >= search_stopping_rule_rep):
                opt_adjusted_cost_diff_list = opt_adjusted_cost_trends_list[(len(opt_adjusted_cost_trends_list)-search_stopping_rule_rep):]
                old_adjusted_cost_value = opt_adjusted_cost_diff_list[0]
                opt_adjusted_cost_diff_list = abs(np.array(opt_adjusted_cost_diff_list) - old_adjusted_cost_value)
                opt_adjusted_cost_diff_list = opt_adjusted_cost_diff_list/(abs(old_adjusted_cost_value)+numerical_precision)
                opt_adjusted_cost_diff_max = max(opt_adjusted_cost_diff_list)
                if opt_adjusted_cost_diff_max <= search_stopping_rule_err:
                    if search_method=="hybrid": ## search_method=="hybrid"
                        search_stopping_rule_counter = 0
                        new_grouping_indexes_list = gen_grouping_indexes_list(N_size, rand=True) ## True: Random grouping, False: Grouping in order
                        if show_info:
                            info_func(info_args, "Grouping has been shuffled.")
                    else: ## search_method=="ex"
                        if show_info:
                            info_func(info_args, "The stopping criterion determined that convergence to the optimum value was achieved.")
                        break
            ## Local grouping: Select two clusters and perform an exchange between the two clusters
            probability_tensor = copy.deepcopy(new_intergroup_weighted_cost_tensor)
            cluster_1_value = (random.choices(probability_tensor, k=1, weights=probability_tensor))[0]
            cluster_1_flattened_index_list = get_tensor_flattened_index_list_from_value(probability_tensor, cluster_1_value, tensor_tolerance)
            cluster_1_flattened_index = random.choice(cluster_1_flattened_index_list)
            cluster_1_multi_index = get_tensor_multi_index_from_flattened_index(cluster_1_flattened_index, N_rank, N_accum)
            probability_tensor[cluster_1_flattened_index] = 0
            cluster_2_value = (random.choices(probability_tensor, k=1, weights=probability_tensor))[0]
            cluster_2_flattened_index_list = get_tensor_flattened_index_list_from_value(probability_tensor, cluster_2_value, tensor_tolerance)
            cluster_2_flattened_index = random.choice(cluster_2_flattened_index_list)
            cluster_2_multi_index = get_tensor_multi_index_from_flattened_index(cluster_2_flattened_index, N_rank, N_accum)
            ## Preparation for local grouping
            local_N_size = []
            local_data_indexes = []
            opt_local_grouping_indexes_list = []
            ## local_N_size, local_data_indexes, opt_local_grouping_indexes_list, local_N_rank, local_N_accum, local_N_size_prod, local_marginal_mass_vectors
            for local_group in range(N_rank):
                if cluster_1_multi_index[local_group] == cluster_2_multi_index[local_group]:
                    local_N_size.append(1)
                    temp_index = new_grouping_indexes_list[local_group][cluster_1_multi_index[local_group]]
                    local_data_indexes.append(temp_index)
                    opt_local_grouping_indexes_list.append([temp_index])
                else:
                    local_N_size.append(2)
                    temp_index_1 = new_grouping_indexes_list[local_group][cluster_1_multi_index[local_group]]
                    temp_index_2 = new_grouping_indexes_list[local_group][cluster_2_multi_index[local_group]]
                    local_data_indexes.append(temp_index_1)
                    local_data_indexes.append(temp_index_2)
                    opt_local_grouping_indexes_list.append([temp_index_1, temp_index_2])
            local_N_size = tuple(local_N_size)
            (local_N_rank, local_N_accum, local_N_size_prod) = get_N(local_N_size)
            local_marginal_mass_vectors = calc_marginal_mass_vectors(local_N_rank, local_N_size)
            ## Calculation of current local optimal transportation costs
            (opt_local_adjusted_cost_value, opt_local_mean_cost_value, opt_local_variance_cost_value,
            opt_local_intragroup_cost_value, opt_local_intragroup_cost_nparray_list, opt_local_intragroup_barycenter_nparray_list,
            opt_local_intergroup_cost_value, opt_local_intergroup_P_tensor, opt_local_intergroup_weighted_cost_tensor,
            opt_local_intergroup_u_vec_list, opt_local_intergroup_f_vec_list,
            opt_local_intergroup_cost_tensor) = calc_adjusted_cost_value(
                opt_local_grouping_indexes_list, data_points_nparray, local_marginal_mass_vectors,
                local_N_size, local_N_rank, local_N_accum, local_N_size_prod,
                mean_penalty_weight, variance_penalty_weight,
                numerical_precision, ot_speed, ot_stopping_rule, ot_loop_max
            )
            old_local_adjusted_cost_value = opt_local_adjusted_cost_value
            ## Enumeration of grouping patterns
            ## (If N_rank is 2 or 3, all enumeration is used, and more than that, random selection is used.)
            local_grouping_indexes_list_combinations = []
            if local_N_rank == 2: ## It might be a good idea to have all the patterns ready in advance. (2^2-1=3)
                numbers_list = list(range(sum(local_N_size)))
                for sub_numbers_list_1 in itertools.combinations(numbers_list, local_N_size[0]):
                    sub_numbers_list_2 = tuple(np.delete(numbers_list, sub_numbers_list_1, 0))
                    temp_local_grouping_indexes_list = list((np.array(local_data_indexes))[list(sub_numbers_list_1+sub_numbers_list_2)])
                    temp_local_grouping_indexes_list = gen_grouping_indexes_list(local_N_size, rand=False, data_order_list=temp_local_grouping_indexes_list)
                    if temp_local_grouping_indexes_list != opt_local_grouping_indexes_list:
                        local_grouping_indexes_list_combinations.append(temp_local_grouping_indexes_list)
            elif local_N_rank == 3: ## It might be a good idea to have all the patterns ready in advance. (2^3-1=7)
                numbers_list = list(range(sum(local_N_size)))
                for sub_numbers_list_1 in itertools.combinations(numbers_list, local_N_size[0]):
                    temp_numbers_list = np.delete(numbers_list, sub_numbers_list_1, 0)
                    for sub_numbers_list_2 in itertools.combinations(temp_numbers_list, local_N_size[1]):      
                        sub_numbers_list_3 = tuple(np.delete(numbers_list, sub_numbers_list_1+sub_numbers_list_2, 0))
                        temp_local_grouping_indexes_list = list((np.array(local_data_indexes))[list(sub_numbers_list_1+sub_numbers_list_2+sub_numbers_list_3)])
                        temp_local_grouping_indexes_list = gen_grouping_indexes_list(local_N_size, rand=False, data_order_list=temp_local_grouping_indexes_list)
                        if temp_local_grouping_indexes_list!= opt_local_grouping_indexes_list:
                                local_grouping_indexes_list_combinations.append(temp_local_grouping_indexes_list)
            else:
                for i in range(local_loop_max):
                    temp_local_grouping_indexes_list = random.sample(local_data_indexes, len(local_data_indexes))
                    temp_local_grouping_indexes_list = gen_grouping_indexes_list(local_N_size, rand=False, data_order_list=temp_local_grouping_indexes_list)
                    if (temp_local_grouping_indexes_list!= opt_local_grouping_indexes_list) and (temp_local_grouping_indexes_list not in local_grouping_indexes_list_combinations):
                                local_grouping_indexes_list_combinations.append(temp_local_grouping_indexes_list)
            ## Calculate the cost of local optimal transportation for each pattern of local grouping
            opt_local_adjusted_cost_value = float('inf')
            opt_local_grouping_indexes_list_list = []
            for new_local_grouping_indexes_list in local_grouping_indexes_list_combinations:
                (new_local_adjusted_cost_value, new_local_mean_cost_value, new_local_variance_cost_value,
                new_local_intragroup_cost_value, new_local_intragroup_cost_nparray_list, new_local_intragroup_barycenter_nparray_list,
                new_local_intergroup_cost_value, new_local_intergroup_P_tensor, new_local_intergroup_weighted_cost_tenso,
                new_local_intergroup_u_vec_list, new_local_intergroup_f_vec_list,
                new_local_intergroup_cost_tensor) = calc_adjusted_cost_value(
                        new_local_grouping_indexes_list, data_points_nparray, local_marginal_mass_vectors,
                        local_N_size, local_N_rank, local_N_accum, local_N_size_prod,
                        mean_penalty_weight, variance_penalty_weight,
                        numerical_precision, ot_speed, ot_stopping_rule, ot_loop_max
                )
                if new_local_adjusted_cost_value < opt_local_adjusted_cost_value:
                    opt_local_adjusted_cost_value = new_local_adjusted_cost_value
                    opt_local_grouping_indexes_list_list = [new_local_grouping_indexes_list]
                elif new_local_adjusted_cost_value == opt_local_adjusted_cost_value:
                    opt_local_grouping_indexes_list_list.append(new_local_grouping_indexes_list)
            opt_local_grouping_indexes_list = random.choice(opt_local_grouping_indexes_list_list)
            random_number = random.random()
            new_grouping_flag = (opt_local_adjusted_cost_value==0) or (random_number <= (old_local_adjusted_cost_value/opt_local_adjusted_cost_value))
            if new_grouping_flag:
                for group in range(local_N_rank):
                    if local_N_size[group] == 1:
                        new_grouping_indexes_list[group][cluster_1_multi_index[group]] = opt_local_grouping_indexes_list[group][0]
                    else:
                        new_grouping_indexes_list[group][cluster_1_multi_index[group]] = opt_local_grouping_indexes_list[group][0]
                        new_grouping_indexes_list[group][cluster_2_multi_index[group]] = opt_local_grouping_indexes_list[group][1]
        if new_grouping_flag:
            ## Calculation of the cost of optimal transport
            (new_adjusted_cost_value, new_mean_cost_value, new_variance_cost_value,
            new_intragroup_cost_value, new_intragroup_cost_nparray_list, new_intragroup_barycenter_nparray_list, 
            new_intergroup_cost_value, new_intergroup_P_tensor, 
            new_intergroup_weighted_cost_tensor, new_intergroup_u_vec_list, new_intergroup_f_vec_list, 
            new_intergroup_cost_tensor) = calc_adjusted_cost_value(
                    new_grouping_indexes_list, data_points_nparray, marginal_mass_vectors, 
                    N_size, N_rank, N_accum, N_size_prod,
                    mean_penalty_weight, variance_penalty_weight,
                    numerical_precision, ot_speed, ot_stopping_rule, ot_loop_max
            )
        if show_info:
            info_func(info_args, "new_grouping_indexes_list: " + str(new_grouping_indexes_list))
            info_func(info_args, "new_adjusted_cost_value: " + str(new_adjusted_cost_value))
        # if drawing_graphs:
        #     (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data_with_patches(is_umap_loaded, 
        #                                                 new_grouping_indexes_list, data_points_nparray, 
        #                                                 N_size, N_rank, N_accum, N_size_prod,
        #                                                 viz2d_x_value, viz2d_y_value, new_intergroup_P_tensor)
        #     # (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, new_grouping_indexes_list, data_points_nparray,
        #     #                             viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(4,3,1), f_title="Mid-calculation")
        if new_adjusted_cost_value <= opt_adjusted_cost_value:
            opt_grouping_indexes_list = copy.deepcopy(new_grouping_indexes_list)
            opt_adjusted_cost_value = new_adjusted_cost_value
            opt_mean_cost_value = new_mean_cost_value
            opt_variance_cost_value = new_variance_cost_value
            opt_intragroup_cost_value = new_intragroup_cost_value
            opt_intragroup_cost_nparray_list = copy.deepcopy(new_intragroup_cost_nparray_list)
            opt_intragroup_barycenter_nparray_list = copy.deepcopy(new_intragroup_barycenter_nparray_list)
            opt_intergroup_cost_value = new_intergroup_cost_value
            opt_intergroup_P_tensor = copy.deepcopy(new_intergroup_P_tensor)
            opt_intergroup_weighted_cost_tensor = copy.deepcopy(new_intergroup_weighted_cost_tensor)
            opt_intergroup_cost_tensor = copy.deepcopy(new_intergroup_cost_tensor)
        ## Recording
        iteration_number_list.append(loop+1)
        elapsed_time = float(time.time() - start_time)
        elapsed_time_list.append(elapsed_time)
        new_adjusted_cost_trends_list.append(new_adjusted_cost_value)
        opt_adjusted_cost_trends_list.append(opt_adjusted_cost_value)
    ## info
    if show_info:
        info_func(info_args, "---------- opt")
        info_func(info_args, "opt_grouping_indexes_list: " + str(init_grouping_indexes_list))
        info_func(info_args, "opt_adjusted_cost_value: " + str(opt_adjusted_cost_value))
        info_func(info_args, "  (opt_intergroup_cost_value, opt_intragroup_cost_value: " + str(opt_intergroup_cost_value) + ", " + str(opt_intragroup_cost_value) + ")")
        info_func(info_args, "  (mean_penalty_weight*opt_mean_cost_value, variance_penalty_weight*opt_variance_cost_value : "
              + str(mean_penalty_weight*opt_mean_cost_value) + ", " + str(variance_penalty_weight*opt_variance_cost_value) + ")")
        ## Computation time
        elapsed_hour = elapsed_time // 3600
        elapsed_minute = (elapsed_time % 3600) // 60
        elapsed_second = (elapsed_time % 3600 % 60)
        info_func(info_args, "computation time:" + str(elapsed_hour).zfill(2) + ":" + str(elapsed_minute).zfill(2) + ":" + str(elapsed_second).zfill(2))
    if drawing_graphs:
        (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data_with_patches(is_umap_loaded, 
                                            opt_grouping_indexes_list, data_points_nparray, 
                                            N_size, N_rank, N_accum, N_size_prod,
                                            viz2d_x_value, viz2d_y_value, opt_intergroup_P_tensor)
        # (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, opt_grouping_indexes_list, data_points_nparray,
        #                             viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(5,4,2), f_title="Optimal value")
        show_P_tensor(opt_intergroup_P_tensor, N_size, N_rank, N_accum, f_size=(4,3), f_title="Optimal value")
     ## return
    return (opt_grouping_indexes_list, opt_intergroup_P_tensor,
            opt_adjusted_cost_value,
            opt_intergroup_cost_value, opt_intragroup_cost_value,
            opt_mean_cost_value, opt_variance_cost_value,
            iteration_number_list, elapsed_time_list,
            new_adjusted_cost_trends_list, opt_adjusted_cost_trends_list,
            viz2d_x_value, viz2d_y_value
            )

## ## data_points_nparray: NumPy array consisting of data points
## N_size: Tuple consisting of the number of elements in each group. If the variable is an integer, the tuple is automatically generated close to equally divided.
## standardization = True ## Standardization
## mean_penalty_weight = 0.2 ## Weight of mean_cost_value
## variance_penalty_weight = 0.8 ## Weight of variance_cost_value
## numerical_precision = 2e-8 ## Values whose absolute value is less than or equal to numerical_precision are treated as 0.
## ot_speed = 0.02 ## Bigger means faster, smaller means stricter
## ot_stopping_rule = 0.02 ## Criteria to stop updating "u". If the relative error of "u" is smaller than the stop criterion, it is terminated.
## ot_loop_max = 200 ## Maximum number of iterations in calc_multi_ot
## tensor_tolerance = 2e-8 ## Tolerance of values when obtaining the tensor index from the value
## global_loop_max = 100 ## Maximum number of iterations in calc_optimal_grouping
## local_loop_max = 100 ## Upper bound on the number of enumerated patterns of local exchange
## init_grouping_indexes_list = None ## If initial value is None, randomly (if init_grouping_rand == True) generates an initial value
## init_grouping_rand = True ## If initial value is None, randomly (if init_grouping_rand == True) generates an initial value
## search_method = "ex" ## "ex": exchange algorithm, "rand": random search, "hybrid": Hybrid of exchange algorithm and random search.
## search_stopping_rule_err = 0.02 ## Criteria to stop searching by exchange algprithm.
## search_stopping_rule_rep = 20 ## It stops when the relative difference in the optimal cost is search_stopping_rule_err or less for search_stopping_rule_rep consecutive periods.
## main_show_info = True ## Flag whether information is displayed or not
## main_drawing_graphs = True ## Flag whether or not to draw graphs
## sub_show_info = False ## Flag whether information is displayed or not
## sub_drawing_graphs = False ## Flag whether or not to draw graphs
## info_func = (lambda info_args, txt: print(str(txt))) ## Function for displaying information
## info_args = None ## Arguments for info_func
## tensor_size_max = 4000 ## Maximum number of elements in the cost tensor. If N_size_prod > tensor_size_max, use an "approximate solution". 
## group_size_max = 20 ## Maximum number of elements to be extracted if the group has a large number of elements. If min(N_size) > group_size_max, use an "approximate solution". 
## loop_max_multiplier = 4 ## Multiplier of the number of loops in the "approximate solution". 
## viz2d_x_value = None ## x-axis values for data visualization (If None, it is automatically calculated.)
## viz2d_y_value = None ## y-axis values for data visualization (If None, it is automatically calculated.)
def gen_optimal_grouping(data_points_nparray, N_size = None, standardization = True,
                           mean_penalty_weight = 0.2, variance_penalty_weight = 0.8, 
                           numerical_precision = 2e-8,
                           ot_speed = 0.02, ot_stopping_rule = 0.02, ot_loop_max = 200,
                           tensor_tolerance = 2e-8, global_loop_max = 100, local_loop_max = 100,
                           init_grouping_indexes_list = None, init_grouping_rand = True,
                           search_method = "ex", search_stopping_rule_err = 0.02, search_stopping_rule_rep = 20,
                           main_show_info = True, main_drawing_graphs = True,
                           sub_show_info = False, sub_drawing_graphs = False,
                           info_func = (lambda info_args, txt: print(str(txt))),
                           info_args = None,
                           tensor_size_max = 4000, group_size_max = 20, loop_max_multiplier = 4,
                           viz2d_x_value = None, viz2d_y_value = None):
    ## N_size
    data_size = len(data_points_nparray)
    if N_size is None:
        info_func(info_args, "Warning: N_size is None.")
        N_size = tuple(data_size)
    if (type(N_size) == int):
        if data_size > N_size:
            (quotient, remainder) = divmod(data_size, N_size)
            N_size = np.full(N_size, quotient)
            for i in range(remainder):
                N_size[i] = N_size[i] + 1
            N_size = tuple(N_size)
        else:
            N_size = tuple(data_size)
    elif (type(N_size) == tuple) or (type(N_size) == list):
        N_size = tuple(N_size)
        if data_size != sum(N_size):
            info_func(info_args, "Warning: The sum of N_size does not match sample size.")
            N_size = tuple(data_size)
    else:
        info_func(info_args, "Warning: N_size must be of type integer or tuple.")
        N_size = tuple(data_size)
    (N_rank, N_accum, N_size_prod) = get_N(N_size)
    res_calc_optimal_grouping = None
    ## Standardization
    if standardization:
        for i in range((data_points_nparray.shape)[1]):
            if np.var(data_points_nparray[:,i]) > 0:
                data_points_nparray[:,i] = (data_points_nparray[:,i] - np.mean(data_points_nparray[:,i]))/np.std(data_points_nparray[:,i])
            else:
                data_points_nparray[:,i] = data_points_nparray[:,i] - np.mean(data_points_nparray[:,i])
    ## Setting Parameters
    if (N_size_prod > tensor_size_max) or (min(N_size) > group_size_max): ## If True, use "approximate solution".
        ## Initial value settings
        if init_grouping_indexes_list is None:
            new_grouping_indexes_list = gen_grouping_indexes_list(N_size, rand=init_grouping_rand) ## True: Random grouping, False: Grouping in order
        else:
            new_grouping_indexes_list = copy.deepcopy(init_grouping_indexes_list)
        if main_show_info:
            info_func(info_args, "---------- new_grouping_indexes_list (initial value): " + str(new_grouping_indexes_list))
        if main_drawing_graphs:
            (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, new_grouping_indexes_list, data_points_nparray,
                                        viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(5,4,2), f_title="Initial value")
        for loop in range( loop_max_multiplier*N_rank ):
            (group_1, group_2) = random.sample(list(range(N_rank)), 2)
            sub_N_size = [N_size[group_1], N_size[group_2]]
            group_1_sub_index = []
            group_2_sub_index = []
            if sub_N_size[0] > group_size_max:
                group_1_sub_index = random.sample(list(range(sub_N_size[0])), group_size_max)
                sub_N_size[0] = group_size_max
            else:
                group_1_sub_index = list(range(sub_N_size[0]))
            if sub_N_size[1] > group_size_max:
                group_2_sub_index = random.sample(list(range(sub_N_size[1])), group_size_max)
                sub_N_size[1] = group_size_max
            else:
                group_2_sub_index = list(range(sub_N_size[1]))
            sub_N_size = tuple(sub_N_size)
            sub_data_index = list(np.array(new_grouping_indexes_list[group_1])[group_1_sub_index]) + list(np.array(new_grouping_indexes_list[group_2])[group_2_sub_index])
            sub_data_points_nparray = data_points_nparray[sub_data_index]
            (sub_N_rank, sub_N_accum, sub_N_size_prod) = get_N(sub_N_size)
            res_calc_optimal_grouping = calc_optimal_grouping(
                sub_data_points_nparray, sub_N_size,
                sub_N_rank, sub_N_accum, sub_N_size_prod,
                mean_penalty_weight, variance_penalty_weight,
                numerical_precision,
                ot_speed, ot_stopping_rule, ot_loop_max,
                tensor_tolerance, global_loop_max, local_loop_max,
                None, True, ## init_grouping_indexes_list, init_grouping_rand,
                search_method, search_stopping_rule_err, search_stopping_rule_rep,
                sub_show_info, sub_drawing_graphs,
                info_func,
                info_args,
                viz2d_x_value, viz2d_y_value)
            sub_opt_grouping_indexes_list = res_calc_optimal_grouping[0]
            group_1_sub_grouping_indexes_list = list(np.array(sub_data_index)[sub_opt_grouping_indexes_list[0]])
            group_2_sub_grouping_indexes_list = list(np.array(sub_data_index)[sub_opt_grouping_indexes_list[1]])
            for i, index in enumerate(group_1_sub_index):
                new_grouping_indexes_list[group_1][index] = group_1_sub_grouping_indexes_list[i]
            for i, index in enumerate(group_2_sub_index):
                new_grouping_indexes_list[group_2][index] = group_2_sub_grouping_indexes_list[i]
            if main_show_info:
                info_func(info_args, "---------- loop (partial optimization): " + str(loop+1))
                info_func(info_args, "---------- new_grouping_indexes_list (partial optimization): " + str(new_grouping_indexes_list))
            if (main_drawing_graphs) and (loop == (2*N_rank-1)):
                (fig, ax, viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, new_grouping_indexes_list, data_points_nparray,
                                            viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(5,4,2), f_title="Optimal value")
        res_calc_optimal_grouping = (new_grouping_indexes_list, 
                                     None, # opt_intergroup_P_tensor,
                                     None, # opt_adjusted_cost_value,
                                     None, # opt_intergroup_cost_value,
                                     None, # opt_intragroup_cost_value,
                                     None, # opt_mean_cost_value,
                                     None, # opt_variance_cost_value,
                                     None, # iteration_number_list,
                                     None, # elapsed_time_list,
                                     None, # new_adjusted_cost_trends_list,
                                     None, # opt_adjusted_cost_trends_list,
                                     viz2d_x_value, viz2d_y_value)
    else:
        res_calc_optimal_grouping = calc_optimal_grouping(data_points_nparray, N_size,
                            N_rank, N_accum, N_size_prod,
                            mean_penalty_weight, variance_penalty_weight,
                            numerical_precision,
                            ot_speed, ot_stopping_rule, ot_loop_max,
                            tensor_tolerance, global_loop_max, local_loop_max,
                            init_grouping_indexes_list, init_grouping_rand,
                            search_method, search_stopping_rule_err, search_stopping_rule_rep,
                            main_show_info, main_drawing_graphs,
                            info_func,
                            info_args,
                            viz2d_x_value, viz2d_y_value)
    ## res_calc_optimal_grouping:
    ## (opt_grouping_indexes_list, opt_intergroup_P_tensor,
    ##  opt_adjusted_cost_value,
    ##  opt_intergroup_cost_value, opt_intragroup_cost_value,
    ##  opt_mean_cost_value, opt_variance_cost_value,
    ##  iteration_number_list, elapsed_time_list,
    ##  new_adjusted_cost_trends_list, opt_adjusted_cost_trends_list,
    ##  viz2d_x_value, viz2d_y_value)
    return res_calc_optimal_grouping

## input_filepath = "./members.csv" ## File path of the input file, in csv format.
## input_index_col = 0 ## Column number with column name or column number in the csv file
## output_filepath = "./grouping.csv" ##  File path of the output file, in csv format.
def gen_grouping_from_csv_file(input_filepath= "./members.csv", input_index_col = 0, output_filepath = "./output_otg.csv",
                           N_size = None,
                           standardization = True,
                           mean_penalty_weight = 0.2, variance_penalty_weight = 0.8, 
                           numerical_precision = 2e-8,
                           ot_speed = 0.02, ot_stopping_rule = 0.02, ot_loop_max = 200,
                           tensor_tolerance = 2e-8, global_loop_max = 100, local_loop_max = 100,
                           init_grouping_indexes_list = None, init_grouping_rand = True,
                           search_method = "ex", search_stopping_rule_err = 0.02, search_stopping_rule_rep = 20,
                           main_show_info = True, main_drawing_graphs = True,
                           sub_show_info = False, sub_drawing_graphs = False,
                           info_func = (lambda info_args, txt: print(str(txt))),
                           info_args = None,
                           tensor_size_max = 4000, group_size_max = 20, loop_max_multiplier = 4,
                           viz2d_x_value = None, viz2d_y_value = None):
    ############################
    ## Loading data: loading csv files
    df = pd.read_csv(filepath_or_buffer=input_filepath, index_col=input_index_col)
    output_data = copy.deepcopy(df)
    data_size = len(df)
    ############################
    ## Dummy variable processing: dummy variable for columns where dtype is object
    df = pd.get_dummies(df, drop_first=True, dtype="float") # float64, uint8, bool
    ############################
    ##  Handling missing values: interpolate by median
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    ############################
    ## data_points_nparray: NumPy array consisting of data points
    data_points_nparray_org = np.array(df.values)
    data_points_nparray = copy.deepcopy(data_points_nparray_org) ## data_points_nparray: NumPy array consisting of data points
    data_points_nparray = data_points_nparray.astype(float)
    ###########################################
    ## Data Standardization
    if standardization:
        for i in range((data_points_nparray.shape)[1]):
            if np.var(data_points_nparray[:,i]) > 0:
                data_points_nparray[:,i] = (data_points_nparray[:,i] - np.mean(data_points_nparray[:,i]))/np.std(data_points_nparray[:,i])
            else:
                data_points_nparray[:,i] = data_points_nparray[:,i] - np.mean(data_points_nparray[:,i])
    ###########################################
    ## Division and Search
    (opt_grouping_indexes_list, opt_intergroup_P_tensor,
     opt_adjusted_cost_value,
     opt_intergroup_cost_value, opt_intragroup_cost_value,
     opt_mean_cost_value, opt_variance_cost_value,
     iteration_number_list, elapsed_time_list,
     new_adjusted_cost_trends_list, opt_adjusted_cost_trends_list,
     viz2d_x_value, viz2d_y_value
    ) = gen_optimal_grouping(data_points_nparray, N_size, standardization,
                            mean_penalty_weight, variance_penalty_weight,
                            numerical_precision,
                            ot_speed, ot_stopping_rule, ot_loop_max,
                            tensor_tolerance, global_loop_max, local_loop_max,
                            init_grouping_indexes_list, init_grouping_rand,
                            search_method, search_stopping_rule_err, search_stopping_rule_rep,
                            main_show_info, main_drawing_graphs,
                            sub_show_info, sub_drawing_graphs,
                            info_func, info_args,
                            tensor_size_max, group_size_max, loop_max_multiplier,
                            viz2d_x_value, viz2d_y_value)
    ###########################################
    ## Output grouping results to csv file
    group_labels_list = np.zeros(data_size)
    group = 0
    for members_list in opt_grouping_indexes_list:
        for member in members_list:
            group_labels_list[member] = int(group)
        group = group + 1
    output_data.insert(loc=0, column="Group", value=group_labels_list.astype(int), allow_duplicates=True)
    if (viz2d_x_value is not None) and (viz2d_y_value is not None):
        output_data.insert(loc=1, column="viz2d_x_value", value=viz2d_x_value.astype(float), allow_duplicates=True)
        output_data.insert(loc=2, column="viz2d_y_value", value=viz2d_y_value.astype(float), allow_duplicates=True)
    output_data.to_csv(output_filepath)
    ###########################################
    ## Return
    return (opt_grouping_indexes_list,
            opt_intergroup_P_tensor,
            opt_adjusted_cost_value,
            opt_intergroup_cost_value, opt_intragroup_cost_value,
            opt_mean_cost_value, opt_variance_cost_value,
            iteration_number_list, elapsed_time_list,
            new_adjusted_cost_trends_list, opt_adjusted_cost_trends_list,
            output_data, viz2d_x_value, viz2d_y_value
    )

############################################################
#### WebUI with Gradio ####

def remove_unnecessary_characters(txt):
    txt = txt.replace(" ", "")
    txt = txt.replace("\n", "")
    txt = txt.replace("\t", "")
    txt = txt.replace(",,", ",")
    txt = txt.replace("[[", "[")
    txt = txt.replace("]]", "]")
    txt = txt.replace("[,[", "[")
    txt = txt.replace("],]", "]")
    txt = txt.removeprefix(",")
    txt = txt.removesuffix(",")
    txt = txt.removeprefix("[")
    txt = txt.removesuffix("]")
    txt = txt.removeprefix(",")
    txt = txt.removesuffix(",")
    return txt

def str_to_int_list(txt):
    txt = remove_unnecessary_characters(txt)
    txt = txt.split(",")
    txt = [int(e) for e in txt]
    return txt

def is_float_str(txt):
    try:
        float(txt)
    except ValueError:
        return False
    else:
        return True
    
def get_cleaned_df_and_viz2d(df, standardization=True):
    df = df.replace("", np.nan).dropna(how='any')
    ## Converts a string that can be converted to a number to a number
    for colname, item in df.items():
        df[colname] = pd.to_numeric(item, errors="ignore")
    df_cleaned = copy.deepcopy(df)
    ## Dummy variable processing: dummy variable for columns where dtype is object
    df_cleaned = pd.get_dummies(df_cleaned, drop_first=True, dtype="float") # float64, uint8, bool
    ##  Handling missing values: interpolate by median
    for col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    ## data_points_nparray: NumPy array consisting of data points
    data_points_nparray = np.array(df_cleaned.values)
    data_points_nparray = data_points_nparray.astype(float)
    ## Data Standardization
    if standardization:
        for i in range((data_points_nparray.shape)[1]):
            if np.var(data_points_nparray[:,i]) > 0:
                data_points_nparray[:,i] = (data_points_nparray[:,i] - np.mean(data_points_nparray[:,i]))/np.std(data_points_nparray[:,i])
            else:
                data_points_nparray[:,i] = data_points_nparray[:,i] - np.mean(data_points_nparray[:,i])
    df_cleaned = pd.DataFrame(data_points_nparray,
                 index=df_cleaned.index.values,
                 columns=df_cleaned.columns.values)
    ## 2d
    viz2d_x_value, viz2d_y_value = gen_2d_data(is_umap_loaded, data_points_nparray)
    ## labels
    viz2d_x_label = "x_umapped"
    viz2d_y_label = "y_umapped"
    if (not(is_umap_loaded)) or (df_cleaned.shape[1] == 2):
        viz2d_x_label = df_cleaned.columns[0]
        viz2d_y_label = df_cleaned.columns[1]
    elif df_cleaned.shape[1] < 2:
        viz2d_x_label = df_cleaned.columns[0]
        viz2d_y_label = ""
    return (df_cleaned, viz2d_x_value, viz2d_y_value, viz2d_x_label, viz2d_y_label)

def load_csv(input_filepath, input_index_col, standardization=True):
    ## Loading data: loading csv file
    df_org = pd.read_csv(filepath_or_buffer=input_filepath, index_col=input_index_col)
    (df_cleaned, viz2d_x_value, viz2d_y_value,
     viz2d_x_label, viz2d_y_label) = get_cleaned_df_and_viz2d(df_org, standardization)
    ## return
    return (df_org, df_cleaned, viz2d_x_value, viz2d_y_value, viz2d_x_label, viz2d_y_label)

def gen_grouping_in_gradio(df_org, df_cleaned, N_size = None,
                           mean_penalty_weight = 0.2, variance_penalty_weight = 0.8, 
                           numerical_precision = 2e-8,
                           ot_speed = 0.02, ot_stopping_rule = 0.02, ot_loop_max = 200,
                           tensor_tolerance = 2e-8, global_loop_max = 100, local_loop_max = 100,
                           init_grouping_index = None, init_grouping_rand = True,
                           search_method = "ex", search_stopping_rule_err = 0.02, search_stopping_rule_rep = 20,
                           main_show_info = True, main_drawing_graphs = False,
                           sub_show_info = False, sub_drawing_graphs = False,
                           info_func = (lambda info_args, txt: print(str(txt))),
                           info_args = None,
                           tensor_size_max = 4000, group_size_max = 20, loop_max_multiplier = 4,
                           viz2d_x_value = None, viz2d_y_value = None):
    data_points_nparray = np.array(df_cleaned.values)
    if search_method == "ex":
        info_func(info_args, "Search for optimal values using the exchange algorithm.") 
    elif search_method == "hybrid":
        info_func(info_args, "Search for optimal values using the hybrid algorithm.")
    else:
        info_func(info_args, "Search for optimal values at random.")
    ## N_size
    data_size = len(data_points_nparray)
    if N_size is None:
        info_func(info_args, "Warning: N_size is None.")
        N_size = tuple(data_size)
    if (type(N_size) == int):
        if data_size > N_size:
            (quotient, remainder) = divmod(data_size, N_size)
            N_size = np.full(N_size, quotient)
            for i in range(remainder):
                N_size[i] = N_size[i] + 1
            N_size = tuple(N_size)
        else:
            N_size = tuple(data_size)
    elif (type(N_size) == tuple) or (type(N_size) == list):
        N_size = tuple(N_size)
        if data_size != sum(N_size):
            info_func(info_args, "Warning: The sum of N_size does not match sample size.")
            N_size = tuple(data_size)
    else:
        info_func(info_args, "Warning: N_size must be of type integer or tuple.")
        N_size = tuple(data_size)
    (N_rank, N_accum, N_size_prod) = get_N(N_size)
    ## Setting Parameters
    res_calc_optimal_grouping = None
    opt_grouping_indexes_list = None
    opt_intergroup_P_tensor = None
    if (N_size_prod > tensor_size_max) or (min(N_size) > group_size_max): ## If True, use "approximate solution".
        ## Initial value settings
        new_grouping_indexes_list = None
        if init_grouping_index is None:
            new_grouping_indexes_list = gen_grouping_indexes_list(N_size, rand=init_grouping_rand) ## True: Random grouping, False: Grouping in order
        else:
            new_grouping_indexes_list = copy.deepcopy(init_grouping_index)
        if main_show_info:
            info_func(info_args, "---------- new_grouping_indexes_list (initial value): " + str(new_grouping_indexes_list))
        if main_drawing_graphs:
            (viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, new_grouping_indexes_list, data_points_nparray,
                                        viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(5,4,2), f_title="Initial value")
        for loop in range( loop_max_multiplier*N_rank ):
            (group_1, group_2) = random.sample(list(range(N_rank)), 2)
            sub_N_size = [N_size[group_1], N_size[group_2]]
            group_1_sub_index = []
            group_2_sub_index = []
            if sub_N_size[0] > group_size_max:
                group_1_sub_index = random.sample(list(range(sub_N_size[0])), group_size_max)
                sub_N_size[0] = group_size_max
            else:
                group_1_sub_index = list(range(sub_N_size[0]))
            if sub_N_size[1] > group_size_max:
                group_2_sub_index = random.sample(list(range(sub_N_size[1])), group_size_max)
                sub_N_size[1] = group_size_max
            else:
                group_2_sub_index = list(range(sub_N_size[1]))
            sub_N_size = tuple(sub_N_size)
            sub_data_index = list(np.array(new_grouping_indexes_list[group_1])[group_1_sub_index]) + list(np.array(new_grouping_indexes_list[group_2])[group_2_sub_index])
            sub_data_points_nparray = data_points_nparray[sub_data_index]
            (sub_N_rank, sub_N_accum, sub_N_size_prod) = get_N(sub_N_size)
            res_calc_optimal_grouping = calc_optimal_grouping(
                sub_data_points_nparray, sub_N_size,
                sub_N_rank, sub_N_accum, sub_N_size_prod,
                mean_penalty_weight, variance_penalty_weight,
                numerical_precision,
                ot_speed, ot_stopping_rule, ot_loop_max,
                tensor_tolerance, global_loop_max, local_loop_max,
                None, True, ## init_grouping_index, init_grouping_rand,
                search_method, search_stopping_rule_err, search_stopping_rule_rep,
                sub_show_info, sub_drawing_graphs,
                info_func, info_args,
                viz2d_x_value, viz2d_y_value)
            sub_opt_grouping_indexes_list = res_calc_optimal_grouping[0]
            group_1_sub_grouping_index = list(np.array(sub_data_index)[sub_opt_grouping_indexes_list[0]])
            group_2_sub_grouping_index = list(np.array(sub_data_index)[sub_opt_grouping_indexes_list[1]])
            for i, index in enumerate(group_1_sub_index):
                new_grouping_indexes_list[group_1][index] = group_1_sub_grouping_index[i]
            for i, index in enumerate(group_2_sub_index):
                new_grouping_indexes_list[group_2][index] = group_2_sub_grouping_index[i]
            if main_show_info:
                info_func(info_args, "---------- loop (partial optimization): " + str(loop+1))
                info_func(info_args, "---------- new_grouping_indexes_list (partial optimization): " + str(new_grouping_indexes_list))
            if (main_drawing_graphs) and (loop == (2*N_rank-1)):
                (viz2d_x_value, viz2d_y_value) = show_2d_data(is_umap_loaded, new_grouping_indexes_list, data_points_nparray,
                                            viz2d_x_value, viz2d_y_value, line_width = 1, f_size=(5,4,2), f_title="Optimal value")
        opt_grouping_indexes_list = new_grouping_indexes_list
    else:
        res_calc_optimal_grouping = calc_optimal_grouping(data_points_nparray, N_size,
                            N_rank, N_accum, N_size_prod,
                            mean_penalty_weight, variance_penalty_weight,
                            numerical_precision,
                            ot_speed, ot_stopping_rule, ot_loop_max,
                            tensor_tolerance, global_loop_max, local_loop_max,
                            init_grouping_index, init_grouping_rand,
                            search_method, search_stopping_rule_err, search_stopping_rule_rep,
                            main_show_info, main_drawing_graphs,
                            info_func, info_args,
                            viz2d_x_value, viz2d_y_value)
        opt_grouping_indexes_list = res_calc_optimal_grouping[0]
        opt_intergroup_P_tensor = res_calc_optimal_grouping[1]
    ## res_calc_optimal_grouping:
    ## (opt_grouping_indexes_list, opt_intergroup_P_tensor,
    ##  opt_adjusted_cost_value,
    ##  opt_intergroup_cost_value, opt_intragroup_cost_value,
    ##  opt_mean_cost_value, opt_variance_cost_value,
    ##  iteration_number_list, elapsed_time_list,
    ##  new_adjusted_cost_trends_list, opt_adjusted_cost_trends_list,
    ##  viz2d_x_value, viz2d_y_value)
    ###########################################
    ## Output grouping results
    output_data = copy.deepcopy(df_org)
    group_labels_list = np.zeros(data_size)
    group = 0
    for members_list in opt_grouping_indexes_list:
        for member in members_list:
            group_labels_list[member] = int(group)
        group = group + 1
    output_data.insert(loc=0, column="Group", value=group_labels_list.astype(int), allow_duplicates=True)
    # output_data.insert(loc=1, column="viz2d_x_value", value=viz2d_x_value.astype(float), allow_duplicates=True)
    # output_data.insert(loc=2, column="viz2d_y_value", value=viz2d_y_value.astype(float), allow_duplicates=True)
    return (output_data, group_labels_list,
            opt_grouping_indexes_list, opt_intergroup_P_tensor,
            N_size, N_rank, N_accum, N_size_prod)

def draw_graph_in_gradio(index_values,
                         viz2d_x_value, viz2d_y_value,
                         viz2d_x_label = "x", viz2d_y_label = "y",
                         group_labels_list = None,
                         grouping_indexes_list = None, patch_weight = None,
                         N_size = None, N_rank = None, N_accum = None, N_size_prod = None):
    fig, ax = plt.subplots()
    ax.cla()
    fig.set_facecolor("#C0C0C0") ## silver=#C0C0C0, lightgray=#D3D3D3, whitesmoke=#F5F5F5, snow=#FFFAFA
    ax.set_facecolor("#F5F5F5") ## silver=#C0C0C0, lightgray=#D3D3D3, whitesmoke=#F5F5F5, snow=#FFFAFA
    ax.set_xlabel(viz2d_x_label)
    ax.set_ylabel(viz2d_y_label)
    patch_weight_dataframe = None
    if group_labels_list is None:
        ax.plot(viz2d_x_value, viz2d_y_value, alpha=0.5, marker="o", markersize=10, linewidth=0)
        for i, lab in enumerate(index_values): ## labels
            ax.annotate(lab, (viz2d_x_value[i], viz2d_y_value[i]))
    else:
        colors = cm.tab10 # cm.tab20
        len_colors = 10
        markers = ["o", "^", "s", "p", "D", "H", "*", "v", "<", ">",  
                        "+", "x", ".", ",", "d", "h", "1", "2", "3", "4", "8", "|", "_"]
        for i, lab in enumerate(index_values):
            p_color = colors(int(group_labels_list[i])%len_colors)
            p_marker = markers[int(group_labels_list[i])%len(markers)]
            ax.plot(viz2d_x_value[i], viz2d_y_value[i],
                            color=p_color, marker=p_marker,
                            alpha=0.5, markersize=10, linewidth=0)
            ax.annotate(index_values[i], (viz2d_x_value[i], viz2d_y_value[i]))
        if patch_weight is not None:
            if (N_rank is None) or (N_accum is None) or (N_size_prod is None):
                (N_rank, N_accum, N_size_prod) = get_N(N_size)
            patch_weight_max = max(patch_weight)
            patch_weight_dataframe_columns = ["weight"]
            for group in range(N_rank):
                patch_weight_dataframe_columns = patch_weight_dataframe_columns + ["label_"+str(group), "x_"+str(group), "y_"+str(group)]
            patch_weight_dataframe = pd.DataFrame(columns=patch_weight_dataframe_columns)
            for m_index in np.ndindex(N_size):
                patch_weight_value = get_tensor_value_from_multi_index(patch_weight, m_index, N_rank, N_accum) / patch_weight_max
                label_vec = []
                x_vec = []
                y_vec = []
                patch_weight_label = "("
                for group in range(N_rank):
                    temp_index = grouping_indexes_list[group][m_index[group]]
                    label_vec.append(index_values[temp_index])
                    x_vec.append(viz2d_x_value[temp_index])
                    y_vec.append(viz2d_y_value[temp_index])
                    patch_weight_label = patch_weight_label + str(index_values[temp_index])
                    if group < (N_rank - 1):
                        patch_weight_label = patch_weight_label + ", "
                    else:
                        patch_weight_label = patch_weight_label + ")"
                label_x_y_vec = []
                for label, x, y in zip(label_vec, x_vec, y_vec):
                    label_x_y_vec = label_x_y_vec + [label, x, y]
                patch_weight_new_row = [patch_weight_value] + label_x_y_vec
                patch_weight_new_dataframe = pd.DataFrame(data=[patch_weight_new_row],
                      index=[patch_weight_label],
                      columns=patch_weight_dataframe.columns)
                if patch_weight_dataframe.empty:
                    patch_weight_dataframe = copy.deepcopy(patch_weight_new_dataframe)
                else:
                    patch_weight_dataframe = pd.concat([patch_weight_dataframe, patch_weight_new_dataframe])
                alp = 0.5 * patch_weight_value / N_rank
                if alp > 0.001:
                    if N_rank > 2:
                        points = get_points_list_in_non_intersecting_order(x_vec, y_vec)
                        patch = patches.Polygon(xy=points, closed=True, alpha=alp, color='black')
                        ax.add_patch(patch)
                    elif N_rank == 2:
                        ax.plot(x_vec, y_vec, alpha=alp, color='black',
                                marker=None, linestyle='solid', linewidth=2)
    plt.close()
    return (fig, ax, patch_weight_dataframe)

class GradioParameters:
    standardization = True
    input_filepath = "./input.csv"
    input_filename = "input.csv"
    input_filedirectory = "./"
    input_index_col = None
    df_org = None
    df_cleaned = None
    df_viz2d = None
    fig = None
    ax = None
    output_filename = "output_otg.csv"
    preprocessed_data_filename = "preprocessed_data.csv"
    viz2d_data_filename = "viz2d_data.csv"
    output_data = None
    group_labels_list = None
    opt_grouping_indexes_list = None
    opt_intergroup_P_tensor = None
    N_size = None ## Tuple consisting of the number of elements in each group. If the variable is an integer, the tuple is automatically generated close to equally divided.
    N_rank = None
    N_accum = None
    N_size_prod = None
    mean_penalty_weight = 0.2 ## Weight of mean_cost_value
    variance_penalty_weight = 0.8 ## Weight of variance_cost_value
    numerical_precision = 2e-8 ## Values whose absolute value is less than or equal to numerical_precision are treated as 0.
    ot_speed = 0.02 ## Bigger means faster, smaller means stricter
    ot_stopping_rule = 0.02 ## Criteria to stop updating "u". If the relative error of "u" is smaller than the stop criterion, it is terminated.
    ot_loop_max = 200 ## Maximum number of iterations in calc_multi_ot
    tensor_tolerance = 2e-8 ## Tolerance of values when obtaining the tensor index from the value
    global_loop_max = 100 ## Maximum number of iterations in calc_optimal_grouping
    local_loop_max = 100 ## Upper bound on the number of enumerated patterns of local exchange
    init_grouping_index = None ## If initial value is None, randomly (if init_grouping_rand == True) generates an initial value
    init_grouping_rand = True ## If initial value is None, randomly (if init_grouping_rand == True) generates an initial value
    search_method = "ex" ## "ex": exchange algorithm, "rand": random search, "hybrid": Hybrid of exchange algorithm and random search.
    search_stopping_rule_err = 0.02 ## Criteria to stop searching by exchange algprithm.
    search_stopping_rule_rep = 20 ## It stops when the relative difference in the optimal cost is search_stopping_rule_err or less for search_stopping_rule_rep consecutive periods.
    main_show_info = True ## Flag whether information is displayed or not
    main_drawing_graphs = False ## Flag whether or not to draw graphs
    sub_show_info = False ## Flag whether information is displayed or not
    sub_drawing_graphs = False ## Flag whether or not to draw graphs
    info_func = (lambda info_args, txt: print(str(txt))) ## Function for displaying information
    info_args = None ## Arguments for info_func
    tensor_size_max = 4000 ## Maximum number of elements in the cost tensor. If N_size_prod > tensor_size_max, use an "approximate solution". 
    group_size_max = 20 ## Maximum number of elements to be extracted if the group has a large number of elements. If min(N_size) > group_size_max, use an "approximate solution". 
    loop_max_multiplier = 4 ## Multiplier of the number of loops in the "approximate solution". 
    viz2d_x_value = None ## x-axis values for data visualization (If None, it is automatically calculated.)
    viz2d_y_value = None ## y-axis values for data visualization (If None, it is automatically calculated.)
    viz2d_x_label = "x"
    viz2d_y_label = "y"
    patch_weight_dataframe = None
    patch_weight_filename = "weight_data_otg.csv"

def upload_file(file, standardization):
    GradioParameters.patch_weight_dataframe = None
    GradioParameters.standardization = standardization
    GradioParameters.input_filepath = file
    GradioParameters.input_index_col = 0
    (GradioParameters.df_org, GradioParameters.df_cleaned,
     GradioParameters.viz2d_x_value, GradioParameters.viz2d_y_value,
     GradioParameters.viz2d_x_label, GradioParameters.viz2d_y_label) = load_csv(
         GradioParameters.input_filepath, GradioParameters.input_index_col, standardization)
    GradioParameters.df_viz2d = pd.DataFrame(data={GradioParameters.viz2d_x_label:GradioParameters.viz2d_x_value,
                                                 GradioParameters.viz2d_y_label:GradioParameters.viz2d_y_value},
                                                 index=GradioParameters.df_org.index.values)
    (GradioParameters.fig, GradioParameters.ax, GradioParameters.patch_weight_dataframe) = draw_graph_in_gradio(GradioParameters.df_org.index.values,
                                               GradioParameters.viz2d_x_value, GradioParameters.viz2d_y_value,
                                               GradioParameters.viz2d_x_label, GradioParameters.viz2d_y_label)
    return ("",
            GradioParameters.df_org.reset_index(),
            GradioParameters.df_cleaned.reset_index(),
            GradioParameters.df_viz2d.reset_index(),
            GradioParameters.fig,
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.Button(value="Hide polygons and line segments", visible=False))

def apply_changes_to_data(members_dataframe, standardization):
    GradioParameters.patch_weight_dataframe = None
    GradioParameters.standardization = standardization
    GradioParameters.df_org = members_dataframe
    GradioParameters.df_org = GradioParameters.df_org.set_index("index")
    GradioParameters.df_org = GradioParameters.df_org.replace("", np.nan).dropna(how='any')
    (GradioParameters.df_cleaned,
     GradioParameters.viz2d_x_value,
     GradioParameters.viz2d_y_value,
     GradioParameters.viz2d_x_label,
     GradioParameters.viz2d_y_label) = get_cleaned_df_and_viz2d(GradioParameters.df_org,
                                                              GradioParameters.standardization)
    GradioParameters.df_viz2d = pd.DataFrame(data={GradioParameters.viz2d_x_label:GradioParameters.viz2d_x_value,
                                                 GradioParameters.viz2d_y_label:GradioParameters.viz2d_y_value},
                                                 index=GradioParameters.df_org.index.values)
    (GradioParameters.fig, GradioParameters.ax, GradioParameters.patch_weight_dataframe) = draw_graph_in_gradio(GradioParameters.df_org.index.values,
                                               GradioParameters.viz2d_x_value, GradioParameters.viz2d_y_value,
                                               GradioParameters.viz2d_x_label, GradioParameters.viz2d_y_label)
    return ("",
            GradioParameters.df_org.reset_index(),
            GradioParameters.df_cleaned.reset_index(),
            GradioParameters.df_viz2d.reset_index(),
            GradioParameters.fig,
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.Button(value="Hide polygons and line segments", visible=False))

def start_grouping(number_of_divisions, speed_slider,
                   m_weight_slider, v_weight_slider,
                   method_dropdown, standardization,
                   global_loop_max_slider, local_loop_max_slider,
                   search_stopping_rule_err_slider, search_stopping_rule_rep_slider,
                   tensor_size_max_slider, group_size_max_slider, loop_max_multiplier_slider,
                   members_dataframe):
    ## Members data
    GradioParameters.patch_weight_dataframe = None
    GradioParameters.standardization = standardization
    GradioParameters.df_org = members_dataframe
    GradioParameters.df_org = GradioParameters.df_org.set_index("index")
    GradioParameters.df_org = GradioParameters.df_org.replace("", np.nan).dropna(how='any')
    (GradioParameters.df_cleaned,
     GradioParameters.viz2d_x_value,
     GradioParameters.viz2d_y_value,
     GradioParameters.viz2d_x_label,
     GradioParameters.viz2d_y_label
     ) = get_cleaned_df_and_viz2d(GradioParameters.df_org, GradioParameters.standardization)
    GradioParameters.df_viz2d = pd.DataFrame(data={GradioParameters.viz2d_x_label:GradioParameters.viz2d_x_value,
                                                 GradioParameters.viz2d_y_label:GradioParameters.viz2d_y_value},
                                                 index=GradioParameters.df_org.index.values)
    ## N_size
    GradioParameters.N_size = str_to_int_list(number_of_divisions)
    if len(GradioParameters.N_size) == 1:
        GradioParameters.N_size = int(GradioParameters.N_size[0])
    GradioParameters.ot_speed = float(speed_slider)
    GradioParameters.mean_penalty_weight = float(m_weight_slider)
    GradioParameters.variance_penalty_weight = float(v_weight_slider)
    GradioParameters.global_loop_max = int(global_loop_max_slider)
    GradioParameters.local_loop_max = int(local_loop_max_slider)
    GradioParameters.search_stopping_rule_err = float(search_stopping_rule_err_slider)
    GradioParameters.search_stopping_rule_rep = int(search_stopping_rule_rep_slider)
    GradioParameters.tensor_size_max = int(tensor_size_max_slider)
    GradioParameters.group_size_max = int(group_size_max_slider)
    GradioParameters.loop_max_multiplier = int(loop_max_multiplier_slider)

    if method_dropdown == "Heuristics":
        GradioParameters.search_method = "ex"
    elif method_dropdown == "Hybrid":
        GradioParameters.search_method = "hybrid"
    else:
        GradioParameters.search_method = "rand"
    (GradioParameters.output_data, GradioParameters.group_labels_list,
         GradioParameters.opt_grouping_indexes_list, GradioParameters.opt_intergroup_P_tensor,
         GradioParameters.N_size, GradioParameters.N_rank,
         GradioParameters.N_accum, GradioParameters.N_size_prod) = gen_grouping_in_gradio(
            GradioParameters.df_org, GradioParameters.df_cleaned,
            GradioParameters.N_size,
            GradioParameters.mean_penalty_weight, GradioParameters.variance_penalty_weight, 
            GradioParameters.numerical_precision,
            GradioParameters.ot_speed, GradioParameters.ot_stopping_rule, GradioParameters.ot_loop_max,
            GradioParameters.tensor_tolerance, GradioParameters.global_loop_max, GradioParameters.local_loop_max,
            GradioParameters.init_grouping_index, GradioParameters.init_grouping_rand,
            GradioParameters.search_method, GradioParameters.search_stopping_rule_err, GradioParameters.search_stopping_rule_rep,
            GradioParameters.main_show_info, GradioParameters.main_drawing_graphs,
            GradioParameters.sub_show_info, GradioParameters.sub_drawing_graphs,
            GradioParameters.info_func,
            None, ## info_args # [GradioParameters.info_address, GradioParameters.info_history],
            GradioParameters.tensor_size_max, GradioParameters.group_size_max, GradioParameters.loop_max_multiplier,
            GradioParameters.viz2d_x_value, GradioParameters.viz2d_y_value)
    (GradioParameters.fig, GradioParameters.ax, GradioParameters.patch_weight_dataframe) = draw_graph_in_gradio(GradioParameters.df_org.index.values,
                                                    GradioParameters.viz2d_x_value,
                                                    GradioParameters.viz2d_y_value,
                                                    GradioParameters.viz2d_x_label,
                                                    GradioParameters.viz2d_y_label,
                                                    GradioParameters.group_labels_list,
                                                    GradioParameters.opt_grouping_indexes_list,
                                                    GradioParameters.opt_intergroup_P_tensor,
                                                    GradioParameters.N_size, GradioParameters.N_rank,
                                                    GradioParameters.N_accum, GradioParameters.N_size_prod)
    if GradioParameters.opt_intergroup_P_tensor is None:
        show_polygons_button = gr.Button(value="", visible=False)
    else:
        show_polygons_button = gr.Button(value="Hide polygons and line segments", visible=True)
    return ("Group index: "+str(GradioParameters.group_labels_list),
            GradioParameters.df_org.reset_index(),
            GradioParameters.df_cleaned.reset_index(),
            GradioParameters.df_viz2d.reset_index(),
            GradioParameters.fig, 
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False),
            show_polygons_button
            )

def export_output_csv_file(e):
    GradioParameters.output_filename = "output_otg." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
    GradioParameters.output_data.to_csv(GradioParameters.output_filename)
    return gr.File(value=GradioParameters.output_filename, visible=True)

def export_preprocessed_data_csv_file(e):
    GradioParameters.preprocessed_data_filename = "preprocessed_data_otg." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
    GradioParameters.df_cleaned.to_csv(GradioParameters.preprocessed_data_filename)
    return gr.File(value=GradioParameters.preprocessed_data_filename, visible=True)

def export_viz2d_data_csv_file(e):
    GradioParameters.viz2d_data_filename = "viz2d_data_otg." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
    GradioParameters.df_viz2d.to_csv(GradioParameters.viz2d_data_filename)
    return gr.File(value=GradioParameters.viz2d_data_filename, visible=True)

def export_patch_weight_data_csv_file(e):
    if GradioParameters.patch_weight_dataframe is not None:
        GradioParameters.patch_weight_filename = "weight_data_otg." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
        GradioParameters.patch_weight_dataframe.to_csv(GradioParameters.patch_weight_filename)
        return gr.File(value=GradioParameters.patch_weight_filename, visible=True)
    else:
        return gr.File(interactive=False, visible=False)

def show_polygons(show_polygons_button):
    if show_polygons_button == "Hide polygons and line segments":
        (GradioParameters.fig, GradioParameters.ax, temp_dataframe) = draw_graph_in_gradio(GradioParameters.df_org.index.values,
                                                    GradioParameters.viz2d_x_value,GradioParameters.viz2d_y_value,
                                                    GradioParameters.viz2d_x_label,GradioParameters.viz2d_y_label,
                                                    GradioParameters.group_labels_list,
                                                    GradioParameters.opt_grouping_indexes_list,
                                                    None,
                                                    GradioParameters.N_size, GradioParameters.N_rank,
                                                    GradioParameters.N_accum, GradioParameters.N_size_prod)
        return (GradioParameters.fig, "Show polygons and line segments")
    else:
        (GradioParameters.fig, GradioParameters.ax, GradioParameters.patch_weight_dataframe) = draw_graph_in_gradio(GradioParameters.df_org.index.values,
                                                    GradioParameters.viz2d_x_value,GradioParameters.viz2d_y_value,
                                                    GradioParameters.viz2d_x_label,GradioParameters.viz2d_y_label,
                                                    GradioParameters.group_labels_list,
                                                    GradioParameters.opt_grouping_indexes_list,
                                                    GradioParameters.opt_intergroup_P_tensor,
                                                    GradioParameters.N_size, GradioParameters.N_rank,
                                                    GradioParameters.N_accum, GradioParameters.N_size_prod)
        return (GradioParameters.fig, "Hide polygons and line segments")

number_of_divisions_info = "Enter an integer greater than 1 or an array consisting of integers"
number_of_divisions_info = number_of_divisions_info + " (Example: entering \"3\" or \"[4,4,4]\" will divide 12 data into three equal parts;"
number_of_divisions_info = number_of_divisions_info + " entering [2,2,8] will create an unbalanced division)"
## Variables
df_init = pd.DataFrame(
(np.array([1,1, 1,2, 1,3, 1,4,  2,1, 2,2, 2,3, 2,4,  3,1, 3,2, 3,3, 3,4,])).reshape(12, 2),
    index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
    columns=["x", "y"])
GradioParameters.df_org = copy.deepcopy(df_init)
(GradioParameters.df_cleaned, 
 GradioParameters.viz2d_x_value,
 GradioParameters.viz2d_y_value, 
 GradioParameters.viz2d_x_label,
 GradioParameters.viz2d_y_label) = get_cleaned_df_and_viz2d(GradioParameters.df_org,
                                                            GradioParameters.standardization)
GradioParameters.df_viz2d = pd.DataFrame(data={GradioParameters.viz2d_x_label:GradioParameters.viz2d_x_value,
                                                 GradioParameters.viz2d_y_label:GradioParameters.viz2d_y_value},
                                           index=GradioParameters.df_org.index.values)
GradioParameters.output_data = copy.deepcopy(GradioParameters.df_org)
(GradioParameters.fig, GradioParameters.ax, GradioParameters.patch_weight_dataframe) = draw_graph_in_gradio(GradioParameters.df_org.index.values,
                                               GradioParameters.viz2d_x_value,
                                               GradioParameters.viz2d_y_value,
                                               GradioParameters.viz2d_x_label,
                                               GradioParameters.viz2d_y_label)

with gr.Blocks(title="Grouping by Optimal Transport") as demo:
    with gr.Row():
        with gr.Column():
            upload_button = gr.UploadButton("Click to Upload a File", file_types=["dataframe"], file_count="single")
            with gr.Group():
                number_of_divisions = gr.Text(value=2, 
                                            label="Number of groups", 
                                            info=number_of_divisions_info,
                                            interactive=True)
                speed_slider = gr.Slider(minimum=0.0001, maximum=1.0, step=0.0001, value=0.02, 
                                    label="Speed", info="Choose between 0.0001 and 1.0", interactive=True)
                m_weight_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.2, 
                                    label="Weight for balance of means",
                                    info="Set weight for balance of means in the range of 0.0 to 1.0", interactive=True)
                v_weight_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, 
                                    label="Weight for balance of variances",
                                    info="Set weight for balance of variances in the range of 0.0 to 1.0", interactive=True)
                method_dropdown = gr.Dropdown(choices=["Heuristics", "Hybrid", "Random search"], value="Heuristics",
                                    label="Search method", info="Choose search method", interactive=True)
                standardization_checkbox = gr.Checkbox(value=True, label="Standardization")
                with gr.Accordion(label="Advanced Settings", open=False):
                    global_loop_max_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=100, 
                                    label="Maximum number of loops in parent process",
                                    info="Set maximum number of loops in parent process in the range of 1 to 10000",
                                    interactive=True)
                    local_loop_max_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=100, 
                                    label="Maximum number of loops in child process",
                                    info="Set maximum number of loops in child process in the range of 1 to 10000",
                                    interactive=True)
                    search_stopping_rule_err_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, value=0.02, 
                                    label="Criteria for stopping rule",
                                    info="Set criteria for stopping rule in the range of 0.0 to 1.0",
                                    interactive=True)
                    search_stopping_rule_rep_slider = gr.Slider(minimum=1, maximum=1000, step=1, value=20, 
                                    label="Number of checks for stopping rule",
                                    info="Set number of checks for stopping rule in the range of 1 to 1000",
                                    interactive=True)
                    tensor_size_max_slider = gr.Slider(minimum=1, maximum=1000000, step=1, value=4000, 
                                    label="Upper limit of tensor size",
                                    info="Set upper limit of tensor size in the range of 1 to 1000000 (If tensor size exceeds upper limit, move to approximate computation)",
                                    interactive=True)
                    group_size_max_slider = gr.Slider(minimum=1, maximum=1000, step=1, value=20, 
                                    label="Upper limit of number of groups",
                                    info="Set upper limit of number of groups in the range of 1 to 1000 (If number of groups exceeds upper limit, move to approximate computation)",
                                    interactive=True)
                    loop_max_multiplier_slider = gr.Slider(minimum=1, maximum=1000, step=1, value=4, 
                                    label="Maximum number of loops in approximate computation",
                                    info="Set maximum number of loops in approximate computation in the range of 1 to 1000",
                                    interactive=True)
                start_button = gr.Button(value="Run")
            with gr.Group():
                progress_log_text = gr.Text(label="Output")
            with gr.Group():
                export_output_button = gr.Button("Export")
                output_csv_file = gr.File(interactive=False, visible=False)
        with gr.Column():
            with gr.Group():
                show_polygons_button = gr.Button(value="", visible=False)
                output_image = gr.Plot(value=GradioParameters.fig)
            with gr.Group():
                apply_changes_to_data_button = gr.Button(value="Apply changes to data")
                members_dataframe = gr.DataFrame(label="Members data",
                                                show_label=True,
                                                value=GradioParameters.df_org.reset_index(),
                                                interactive=True,
                                                height=300)
            with gr.Group():
                preprocessed_dataframe = gr.DataFrame(label="Preprocessed data",
                                                show_label=True,
                                                value=GradioParameters.df_cleaned.reset_index(),
                                                interactive=False,
                                                height=300)
                export_preprocessed_data_button = gr.Button("Export \"Preprocessed data\"")
                preprocessed_data_csv_file = gr.File(interactive=False, visible=False)
            with gr.Group():
                viz2d_dataframe = gr.DataFrame(label="Data used for 2D visualization",
                                                show_label=True,
                                                value=GradioParameters.df_viz2d.reset_index(),
                                                interactive=False,
                                                height=300)
                export_viz2d_data_button = gr.Button("Export \"Data used for 2D visualization\"")
                viz2d_data_csv_file = gr.File(interactive=False, visible=False)
            with gr.Group():
                export_patch_weight_data_button = gr.Button("Export \"Weights for polygons and line segments\"")
                patch_weight_data_csv_file = gr.File(interactive=False, visible=False)
    upload_button.upload(fn=upload_file,
                            inputs=[upload_button, standardization_checkbox],
                            outputs=[progress_log_text,
                                     members_dataframe, preprocessed_dataframe, viz2d_dataframe,
                                     output_image,
                                     output_csv_file,
                                     preprocessed_data_csv_file,
                                     viz2d_data_csv_file,
                                     patch_weight_data_csv_file,
                                     show_polygons_button])
    apply_changes_to_data_button.click(fn=apply_changes_to_data,
                                       inputs=[members_dataframe, standardization_checkbox],
                                       outputs=[progress_log_text,
                                                members_dataframe, preprocessed_dataframe, viz2d_dataframe,
                                                output_image,
                                                output_csv_file,
                                                preprocessed_data_csv_file,
                                                patch_weight_data_csv_file,
                                                viz2d_data_csv_file,
                                                show_polygons_button])
    start_button.click(fn=start_grouping,
                        inputs=[number_of_divisions, speed_slider, 
                                m_weight_slider, v_weight_slider, 
                                method_dropdown, standardization_checkbox,
                                global_loop_max_slider, local_loop_max_slider,
                                search_stopping_rule_err_slider, search_stopping_rule_rep_slider,
                                tensor_size_max_slider, group_size_max_slider, loop_max_multiplier_slider,
                                members_dataframe],
                        outputs=[progress_log_text,
                                 members_dataframe, preprocessed_dataframe, viz2d_dataframe,
                                 output_image,
                                 output_csv_file,
                                 preprocessed_data_csv_file,
                                 viz2d_data_csv_file,
                                 patch_weight_data_csv_file,
                                 show_polygons_button])
    export_output_button.click(fn=export_output_csv_file, inputs=[export_output_button], outputs=[output_csv_file])
    export_preprocessed_data_button.click(fn=export_preprocessed_data_csv_file, inputs=[export_preprocessed_data_button], outputs=[preprocessed_data_csv_file])
    export_viz2d_data_button.click(fn=export_viz2d_data_csv_file, inputs=[export_viz2d_data_button], outputs=[viz2d_data_csv_file])
    export_patch_weight_data_button.click(fn=export_patch_weight_data_csv_file, inputs=[export_patch_weight_data_button], outputs=[patch_weight_data_csv_file])
    show_polygons_button.click(fn=show_polygons,
                                 inputs=[show_polygons_button],
                                 outputs=[output_image, show_polygons_button])
    
demo.css = "footer {visibility: hidden}"
demo.launch(inbrowser=True, show_api=False, favicon_path="favicon.ico")
