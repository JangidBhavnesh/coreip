import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Not the cleanest way of doing this, yet.
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
    with open('../../src/orbitalenergy.json', 'r') as f:
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
    # Input arguments
    in_filename = 'graph_data.json'
    if len(sys.argv) > 1:
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
    test_error = []
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

        predict, errors = test_model_p2(weights, element_list, lemcmat, ref_energies, exp_energies)
        predict_loss = np.sqrt(np.mean((errors) ** 2))
        test_loss = [predict_loss]
        test_error.append(errors)
        print(f"CV #{n_cv}: Testing RMSE over {np.sum(~mask)} samples: {predict_loss:.3f}eV")

    print(f"Average {num_cross_val}-fold CV Training RMSE: {np.mean(train_loss)}eV")
    print(f"Average {num_cross_val}-fold CV Testing RMSE: {np.mean(test_loss)}eV")
    

    test_error = np.array(test_error)

    # Error statisitcs
    mae = np.mean([np.mean(np.abs(er)) for er in test_error])
    stdev = np.mean([np.std(er) for er in test_error])
    rmse = np.mean([np.sqrt(np.mean(er**2)) for er in test_error])
    mean_error = np.mean([np.mean(er) for er in test_error])
    max_error = np.max([np.max(np.abs(er)) for er in test_error])

    stats = {
        'MAE': mae,
        'STDEV': stdev,
        'RMSE': rmse,
        'MSE': mean_error, # Mean-Signed Error
        'Max Error': max_error}

    print("\nError Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # # Plotting the erros
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.bar(stats.keys(), stats.values())
    # ax.set_ylabel('Error Value')
    # ax.set_title('Error Statistics')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()
    '''
    CV #0: Training RMSE over 3894 samples: 3.112eV
    CV #0: Testing RMSE over 1298 samples: 3.924eV
    CV #1: Training RMSE over 3894 samples: 3.236eV
    CV #1: Testing RMSE over 1298 samples: 6.301eV
    CV #2: Training RMSE over 3894 samples: 3.281eV
    CV #2: Testing RMSE over 1298 samples: 3.406eV
    CV #3: Training RMSE over 3894 samples: 3.113eV
    CV #3: Testing RMSE over 1298 samples: 4.059eV
    Average 4-fold CV Training RMSE: 3.1855371045349883eV
    Average 4-fold CV Testing RMSE: 4.0588858996889075eV

    Error Statistics:
    MAE: 1.6177
    STDEV: 4.4216
    RMSE: 4.4223
    MSE: 0.0618
    Max Error: 170.0104
    '''