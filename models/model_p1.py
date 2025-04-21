import sys
import json
import networkx as nx
import numpy as np
from src.smiletovectors import get_full_neighbor_vectors
from src.paulingelectro import get_eleneg_diff_mat
from scipy import optimize
'''
This file provides functionality for fitting to the function:
y = mx

y := experimental binding energy - atomic orbital energy
m := weights to train
x := electronegativity environment 
'''


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

def get_l(l):
    '''
    l = s, p, d, f, g
    '''
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g':4,}.get(l, "l value is not valid")

def get_n_l(orb):
    n = int(orb[0])
    l = get_l(str(orb[1]))
    return n, l

def giveorbitalenergy(ele, orb, orbital_energy_file='orbitalenergy.json'):
    '''
    For a given element and orbital, return the orbital energy in eV
    '''
    with open(orbital_energy_file, 'r') as f:
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

def process_nodes(node_raw_data, vectors, en_mat):
    node_data = []
    assert len(vectors) == len(node_raw_data), f'{len(vectors)} != {len(node_raw_data)}'
    for node_idx, data in enumerate(zip(vectors, node_raw_data)):
        v, n = data
        idx, symbol, cmat = v
        
        lmat = np.zeros((1,en_mat.shape[0]))
        lmat[0, Chem.Atom(symbol).GetAtomicNum() - 1] = 1
        
        e_neg_score = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)[0]
        
        atom_type = n['atom_type']
        formal_charge = n['formal_charge']
        
        for orb, binding_e in zip(n['orbitals'], n['binding_energies']):
            if orb == -1 or binding_e == -1:
                continue
            ref_e = -giveorbitalenergy(symbol, orb)
            n, l = get_n_l(orb)
            assert symbol == atom_type
            # Binding energy zeroed against reference energy
            # AKA, train against the difference from reference atomic orbital energy
            # Each entry is as follows:
            # [Atomic Num, formal charge, e_neg_score, quantum number 'n', quantum number 'l', atomic_orbital_e, binding_e]
            node_data += [np.array([Chem.Atom(symbol).GetAtomicNum(), formal_charge, e_neg_score, n, l, ref_e, binding_e],dtype=np.float64)]
    return node_data

def networkx2arr(data, num_elements=100):
    smiles = list(data.keys())
    en_mat = get_eleneg_diff_mat(num_elements)

    graph_data = []
    for graph_num, smile in enumerate(smiles):
        graph = data[smile]
        print(f"Processing graph {graph_num+1}/{len(smiles)}: {smile}")
        
        try:
            vectors = get_full_neighbor_vectors(smile)
        except:
            print(f"Skipping smile {smile} due to RDKit error")
            continue
        nodes, node_raw_data = zip(*graph.nodes(data=True))
        # Hydrogen will never have a core binding energy
        # For atom-centric data formatting, Hydrogen may be safely omitted
        temp_list = []
        for n in node_raw_data:
            if n['atom_type'] == 'H':
                continue
            temp_list += [n]
        node_raw_data = tuple(temp_list)
        
        ### Process Node information
        new_data = process_nodes(node_raw_data, vectors, en_mat)
        if len(new_data) != 0:
            graph_data += [np.atleast_2d(new_data)]
        else:
            print(f"Found no information in {smile}")

    return np.vstack(graph_data)

def get_xvec(weights, element_list):
    return np.array([weights[int(el)-1] for el in element_list])

def test_model_p1(weights, elements, e_neg, ref_e, exp_e):
    '''
    Return the testing loss for Polynomial Model 1 with weights 'weights'
        - model_p1: Fit the function y=mx
        - y = exp_e - ref_e
        - m = weights
        - x = electronegativity environment 'e_neg'
    Arguments:
        - weights: 1D array of length 'num_element'
        - elements: 1D array of atomic numbers
        - e_neg: 1D array of electronegativity environments
        - ref_e: Atomic orbital energy (reference)
        - exp_e: Experimental binding energy
    Returns:
        predict: Predictions
        loss: RMSE Error
    '''        
    xvec = get_xvec(weights, elements)
    predict = e_neg * xvec + ref_e
    loss = np.sqrt(np.mean((predict - exp_e) ** 2))
    return predict, loss


def train_model_p1(elements, e_neg, ref_e, exp_e, num_elements=100):
    '''
    Return the optimized weights for fitting Polynomial Model 1
        - model_p1: Fit the function y=mx
        - y = exp_e - ref_e
        - m = weights
        - x = electronegativity environment 'e_neg'
    Arguments:
        - elements: 1D array of atomic numbers
        - e_neg: 1D array of electronegativity environments
        - ref_e: Atomic orbital energy (reference)
        - exp_e: Experimental binding energy
        - num_elements: Number of elements to compute weights for (fields will be 0 for elements not encountered in training)
    Returns:
        weights: 1D array of length 'num_element'
    '''
    weights = np.zeros(num_elements)
    def errorfunc(x):
        xvec = get_xvec(x, elements)
        loss = np.sqrt(np.mean((e_neg * xvec + ref_e - exp_e) ** 2))
        return loss
    
    results = optimize.minimize(errorfunc, weights)
    weights = results['x']
    loss = results['fun']
    return weights, loss

from rdkit import Chem
if __name__=='__main__':
    # Input arguments
    in_filename = sys.argv[1] # networkx .json file
    out_filename = 'graphs.pt' # File to save pyTorch graphs
    if len(sys.argv) > 2:
        out_filename = sys.argv[2]
    
    # Load in networkx graphs
    data = load_data_from_file(in_filename)
    num_elements = 100
    # [Atomic Num, formal charge, e_neg_score, quantum number 'n', quantum number 'l', atomic_orbital_e, binding_e]
    graph_data = networkx2arr(data, num_elements)
    
    null_loss = np.sqrt(np.mean((graph_data[:,5] - graph_data[:,6])**2))
    print(f"Null Loss: {null_loss:.3f}eV")

    train_samples = 2500
    test_samples = graph_data.shape[0] - train_samples
    
    ### Training
    exp_energies = graph_data[:train_samples,6]
    ref_energies = graph_data[:train_samples, 5]
    lemcmat = graph_data[:train_samples, 2]
    element_list = graph_data[:train_samples, 0].flatten()
    
    weights, loss = train_model_p1(element_list, lemcmat, ref_energies, exp_energies)
    print(f"Training RMSE over {train_samples} samples: {loss:.3f}eV")
    
    ### Testing
    exp_energies = graph_data[train_samples:train_samples+test_samples, 6]
    ref_energies = graph_data[train_samples:train_samples+test_samples, 5]
    exp_minus_ref = exp_energies - ref_energies
    element_list = graph_data[train_samples:train_samples+test_samples, 0]
    lemcmat = graph_data[train_samples:train_samples+test_samples, 2]

    predict, predict_loss = test_model_p1(weights, element_list, lemcmat, ref_energies, exp_energies)
    
    print(f"Testing RMSE over {test_samples} samples: {predict_loss:.3f}eV")
