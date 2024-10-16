import numpy as np
from typing import Dict
import networkx as nx
import random


def scm_bivariate_continuous(num_env: int, num_sample: int) -> Dict:
    """
    Create Binary Exchangeable Data for Bivariate Graph
    :param seed:
    :param num_env:
    :return:
    """

    # Instantiating the output results: data
    data = {}

    num_var = 2
    # todo: change this for cause/mechanism variability
    # collapse the num_env dimension and make that data a copy
    exchg_mode:int = 0

    # here N[0,:] is theta, N[1,:] is psi, N is the concatenation of the two, and sample from a prior which is uniform[-1, 1]. 
    # Then Nprime is a sample based on identical N for each environment that simulates p(xi, yi | theta, psi). 
    # and xi, yi is the a functional mapping ANprime where A determines the causation direction.
    N = np.random.uniform(-1, 1, (num_var, num_env))
    N = np.repeat(N[:, :, np.newaxis], num_sample, axis=2)
    Nprime = np.random.laplace(N)

    # Define a switch variable to determine the sampling mode
    switch_var = np.random.choice([0, 1, 2])
    
    # Create a new N array with the same shape as before
    N_new = np.zeros_like(N)
    
    if switch_var == 0:
        # Both N[0,:] and N[1,:] are sampled from uniform
        N_new = N
    elif switch_var == 1:
        # Only N[0,:] is random, N[1,:] is constant across environments
        N_new[0, :, :] = N[0, :, :]
        N_new[1, :, :] = N[1, 0, :][:, np.newaxis]
    else:  # switch_var == 2
        # Only N[1,:] is random, N[0,:] is constant across environments
        N_new[0, :, :] = N[0, 0, :][:, np.newaxis]
        N_new[1, :, :] = N[1, :, :]
    
    # Replace the original N with the new N
    N = N_new
    
    

    mode = np.random.choice(3)
    if mode == 0:
        A = np.tril(np.random.randint(1, 10, size=(num_var, num_var)), 0)
        A[0, 0] = 1
        data['true_structure'] = set([(0, 1)])

    elif mode == 1:
        A = np.triu(np.random.randint(1, 10, size=(num_var, num_var)), 0)
        A[1, 1] = 1
        data['true_structure'] = set([(1, 0)])
    elif mode == 2:
        A = np.eye(2)
        data['true_structure'] = set([])

    D = np.einsum('ij, jkh->ikh', A, Nprime)


    if exchg_mode ==0: #CdF
        pass
    elif exchg_mode ==1: #cause var
        pass
        # collapse the cause dimension for envs
    elif exchg_mode ==2: #mech var
        pass
        

    # todo: I need to understand this part to figure out how the data is generated
    env_idx = np.random.choice(num_env, int(num_env/2), replace = False)
    env_mask = np.zeros((num_var, num_env, num_sample), dtype = bool)
    env_mask[:, env_idx, :] = True
    B = (A - np.eye(num_var))*np.random.randint(1, 10)
    D += np.einsum('ij, jkh->ikh', B, Nprime**2) * env_mask


    Data = D.reshape(num_var, -1).T
    c_indx = np.repeat(range(1, num_env + 1), num_sample).reshape(-1, 1).astype(float)

    X = D[0, :, :2] #num_env, num_sample
    Y = D[1, :, :2]
    data['causal-de-finetti'] = {
        'x': X,
        'y': Y,
    }
    data['cd-nod'] = {
        'data': Data,
        'c_indx': c_indx,
    }
    data['x'] = X
    data['y'] = Y

    return data


def scm_multivariate_binary(num_env: int, num_sample: int, num_var: int) -> Dict:
    """
    Create Binary Exchangeable Data for Bivariate Graph
    :param seed:
    :param num_env:
    :return:
    """

    # Instantiating the output results: data
    data = {}

    alpha = 1
    beta = 3

    # Ensure there are specified num_var in the generated DAG
    nodes_generated = []
    while len(nodes_generated) != num_var or len(nodes_generated) == 0:
        G = nx.gnp_random_graph(num_var, 0.5, directed=True)
        DAG = nx.DiGraph([(u, v, {'weight': random.randint(-10, 10)}) for (u, v) in G.edges() if u < v])
        nodes_generated = DAG.nodes

    thetas = np.random.beta(alpha, beta, (num_env, num_var))
    # Generate Data
    Data = np.zeros((num_sample*num_env, num_var))
    var_dict = {str(x): None for x in range(num_var)}
    var_dict['0'] = np.random.binomial(1, p = thetas[:, 0], size = (num_sample, num_env))
    Data[:, 0] = var_dict['0'].T.reshape(-1)
    for d in range(1, num_var):
        factor = np.random.binomial(1, p = thetas[:, d], size = (num_sample, num_env))
        predecessors = [x for x in DAG.predecessors(d)]
        if predecessors != []:
            multi = 1
            for x in DAG.predecessors(d):
                multi *= var_dict[str(x)]
            var_dict[str(d)] = (factor != multi).astype(int)
        else:
            var_dict[str(d)] = factor
        Data[:, d] = var_dict[str(d)].T.reshape(-1)

    data['true_structure'] = set(DAG.edges)
    data['data'] = var_dict

    Data = Data.astype(int)
    c_indx = np.repeat(range(1, num_env + 1), num_sample).reshape(-1, 1).astype(float)
    data['cd-nod'] = {
        'data': Data,
        'c_indx': c_indx,
        'true_structure': set(DAG.edges)
    }

    return data

