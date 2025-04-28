import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Not the cleanest way of doing this, yet.
import json
import networkx as nx
import numpy as np
from src.smiletovectors import get_full_neighbor_vectors
from src.paulingelectro import get_eleneg_diff_mat
from scipy import optimize
from rdkit import Chem
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

'''
This file provides functionality for fitting to the function:
y = ax^n + bx^n-1 + cx^n-2 .....+z

y := experimental binding energy - atomic orbital energy
x := electronegativity environment 
a, b, c... := weights to train
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

def giveorbitalenergy(ele, orb, orbital_energy_file='../../src/orbitalenergy.json'):
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

def process_nodes(data, smile, en_mat):
    '''
    Provided the raw data for a node into a matrix form with our desired features/labels
    Arguments:
        - data: networkx graph
        - smile: SMILE string to use as key in 'data'
        - en_mat: electronegativity difference matrix
    returns:
        matrix of size n_samples X n_features+n_labels
    '''
    graph = data[smile]    
    try:
        vectors = get_full_neighbor_vectors(smile)
    except:
        print(f"Skipping smile {smile} due to RDKit error")
        return None
    
    nodes, node_raw_data = zip(*graph.nodes(data=True))
    node_data = []
    
    # Hydrogen will never have a core binding energy
    # For atom-centric data formatting, Hydrogen may be safely omitted
    temp_list = []
    for n in node_raw_data:
        if n['atom_type'] == 'H':
            continue
        temp_list += [n]
    node_data_no_H = tuple(temp_list)
    
    assert len(vectors) == len(node_data_no_H), f'{len(vectors)} != {len(node_data_no_H)}'
    for node_idx, data in enumerate(zip(vectors, node_data_no_H)):
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
        print(f"Processing graph {graph_num+1}/{len(smiles)}: {smile}")
        ### Process Node information
        new_data = process_nodes(data, smile, en_mat)
        if new_data is None:
            continue
        
        if len(new_data) != 0:
            graph_data += [np.atleast_2d(new_data)]
        else:
            print(f"Found no information in {smile}")

    return np.vstack(graph_data)

def get_xvec(weights, element_list):
    return np.array([weights[int(el)-1] for el in element_list])

def test_model_pn(model, elements, e_neg, ref_e, exp_e):
    xvec = np.array([ele for ele in elements])
    x = e_neg * xvec
    y = ref_e - exp_e
    y_pred = model.predict(x.reshape(-1, 1))
    error = y_pred - y
    return y_pred, error


def train_model_pn(elements, e_neg, ref_e, exp_e, degree=1, num_elements=100):
    xvec = np.array([ele for ele in elements])
    x = e_neg * xvec
    y = ref_e - exp_e
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression(fit_intercept=True))
    # model = make_pipeline(
    # StandardScaler(),  # scaling helps SGD
    # PolynomialFeatures(degree=1),
    # SGDRegressor(loss='squared_error', learning_rate='constant', 
    #              eta0=0.01, max_iter=1000, tol=1e-6, verbose=1))
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    rmse = mean_squared_error(y, y_pred)
    print(f"Training MSE: {rmse:.5f}")
    return model, rmse

if __name__=='__main__':
    # Input arguments
    in_filename = 'graph_data.json'
    if len(sys.argv) > 1:
        in_filename = sys.argv[1] # networkx .json file
    out_filename = 'graphs.pt' # File to save pyTorch graphs
    if len(sys.argv) > 2:
        out_filename = sys.argv[2]
    degree = 1

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
    
    model, loss = train_model_pn(element_list, lemcmat, ref_energies, exp_energies, degree=1)
    print(f"Training RMSE over {train_samples} samples: {loss:.3f}eV")
    
    ### Testing
    exp_energies = graph_data[train_samples:train_samples+test_samples, 6]
    ref_energies = graph_data[train_samples:train_samples+test_samples, 5]
    exp_minus_ref = exp_energies - ref_energies # It should be other way
    element_list = graph_data[train_samples:train_samples+test_samples, 0]
    lemcmat = graph_data[train_samples:train_samples+test_samples, 2]

    predict, errors = test_model_pn(model, element_list, lemcmat, ref_energies, exp_energies)
    print(f"Testing RMSE over {test_samples} samples: {np.sqrt(np.mean((errors) ** 2)):.3f}eV")

   # Error statisitcs
    mae = np.mean(np.abs(errors))
    stdev = np.std(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(np.abs(errors))

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



