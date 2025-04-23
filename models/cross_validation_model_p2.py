import sys
import json
import networkx as nx
import numpy as np
from src.smiletovectors import get_full_neighbor_vectors
from src.paulingelectro import get_eleneg_diff_mat
import matplotlib.pyplot as plt
from scipy import optimize
from rdkit import Chem
from model_p2 import *
au2eV = 27.21139

### Helper functions from hackathon
def load_data_from_file(filename) -> dict:
    """
    Load a dictionary of graphs from JSON file.
    """
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
    return _load_data_from_string_dict(string_dict)

def _load_data_from_string_dict(string_dict) -> dict:
	result_dict = {}
	for key in string_dict:
		graph = nx.node_link_graph(string_dict[key], edges="edges")
		result_dict[key] = graph
	return result_dict

### Custom functions to get orbital energies of lone atoms
def get_l(l):
    '''
    l = s, p, d, f, g
    '''
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g':4,}.get(l, "l value is not valid")

def get_n_l(orb):
    n = int(orb[0])
    l = get_l(str(orb[1]))
    return n, l

def giveorbitalenergy(ele, orb):
    with open('orbitalenergy.json', 'r') as f:
        data = json.load(f)
    try:
        orbenegele = data[ele]
        del data
    except KeyError:
        raise KeyError("Element symbol not found")
    
    n, l = get_n_l(orb)
    cbenergy = orbenegele[str(l)][n-l-1]
    cbenergy *= au2eV
    return cbenergy

if __name__=='__main__':
    np.random.seed(42)
    in_filename = sys.argv[1] # networkx .json file
    out_filename = 'graphs.pt' # File to save pyTorch graphs
    if len(sys.argv) > 2:
        out_filename = sys.argv[2]
    
    # Load in networkx graphs
    data = load_data_from_file(in_filename)
    num_elements = 100
    # [Atomic Num, formal charge, e_neg_score, quantum number 'n', quantum number 'l', atomic_orbital_e, binding_e]
    graph_data = networkx2arr(data, num_elements)
    
    def get_xvec(weights, element_list):
        return np.array([weights[el] for el in element_list])
    
    num_total_data = graph_data.shape[0]
    num_cross_val = 4
    subset_size = num_total_data // num_cross_val
    
    train_loss = []
    test_loss = []
    
    indices = np.arange(num_total_data)
    np.random.shuffle(indices)
    subset_idx = np.random.permutation(num_total_data)[:(num_cross_val*subset_size)]
    for n_cv, idx in enumerate(subset_idx.reshape((num_cross_val, subset_size))):
        mask = np.ones(len(indices), dtype=bool)
        mask[idx] = False
        train_idx = indices[mask]
        test_idx = indices[~mask]
        
        exp_energies = graph_data[train_idx,6]
        ref_energies = graph_data[train_idx, 5]
        lemcmat = graph_data[train_idx, 2]
        element_list = graph_data[train_idx, 0].flatten()

        weights, loss = train_model_p2(element_list, lemcmat, ref_energies, exp_energies)
        with open(f"model_p2_CV{n_cv}_weights.txt", 'w') as f:
            f.write(str(weights.reshape(-1,2)))
            
        train_loss += [loss]
        print(f"CV #{n_cv}: Training RMSE over {np.sum(mask)} samples: {loss:.3f}eV")

        exp_energies = graph_data[test_idx, 6]
        ref_energies = graph_data[test_idx, 5]
        exp_minus_ref = exp_energies - ref_energies
        element_list = graph_data[test_idx, 0]
        lemcmat = graph_data[test_idx, 2]

        predict, predict_loss = test_model_p2(weights, element_list, lemcmat, ref_energies, exp_energies)
        test_loss = [predict_loss]
        print(f"CV #{n_cv}: Testing RMSE over {np.sum(~mask)} samples: {predict_loss:.3f}eV")

    print(f"Average {num_cross_val}-fold CV Training RMSE: {np.mean(train_loss)}eV")
    print(f"Average {num_cross_val}-fold CV Testing RMSE: {np.mean(test_loss)}eV")
    
