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
    smiles = list(data.keys())
    en_mat = get_eleneg_diff_mat()
    
    exp_energies = []
    ref_energies = []
    cmat = []
    lmat = []
    for smile in smiles:
        try:
            vectors = get_full_neighbor_vectors(smile)
            nodes, node_raw_data = zip(*data[smile].nodes(data=True))
            
            # Hydrogen will never have a core binding energy
            # For atom-centric data formatting, Hydrogen may be safely omitted
            temp_list = []
            for n in node_raw_data:
                if n['atom_type'] == 'H':
                    continue
                temp_list += [n]
            node_raw_data = tuple(temp_list)
        except:
            print(f"Skipping smile {smile} due to RDKit error")
            continue
            
        for v, n in zip(vectors, node_raw_data):
            idx, symbol, vec = v
            for orb, binding_e in zip(n['orbitals'], n['binding_energies']):
                if orb == -1 or binding_e == -1:
                    continue

                temp_lmat = np.zeros(en_mat.shape[0])
                temp_lmat[Chem.Atom(symbol).GetAtomicNum() - 1] = 1
                
                temp_cmat = [vec]
                
                lmat += [temp_lmat]
                cmat += [temp_cmat]
                exp_energies += [binding_e]
                orb_en = -giveorbitalenergy(symbol, orb)
                ref_energies += [orb_en]

    full_lmat = np.array(lmat)
    full_lmat = np.atleast_2d(full_lmat)
    full_cmat = np.array(cmat).squeeze().T
    full_cmat = full_cmat.reshape(full_lmat.T.shape)
    full_exp_energies = np.array(exp_energies)
    full_ref_energies = np.array(ref_energies)
    
    null_loss = np.sqrt(np.mean((full_ref_energies - full_exp_energies)**2))
    print(f"Null Loss: {null_loss:.3f}eV")
    
    def get_xvec(weights, element_list):
        return np.array([weights[el] for el in element_list])
    
    num_total_data = len(full_lmat)
    num_cross_val = 8
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
        
        lmat = full_lmat[train_idx, :]
        cmat = full_cmat[:, train_idx]
        exp_energies = full_exp_energies[train_idx]
        ref_energies = full_ref_energies[train_idx]
        lemcmat = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)
        element_list = lmat.argmax(axis=1)
        weights = np.zeros(cmat.shape[0])

        def errorfunc(x):
            xvec = get_xvec(x, element_list)
            loss = np.sqrt(np.mean((lemcmat * xvec + ref_energies - exp_energies) ** 2))
            return loss
    
        results = optimize.minimize(errorfunc, weights)
        weights = results['x']
        train_loss += [results['fun']]
        print(f"CV #{n_cv}: Training RMSE over {np.sum(mask)} samples: {results['fun']:.3f}eV")

        lmat = full_lmat[test_idx, :]
        cmat = full_cmat[:, test_idx]
        exp_energies = full_exp_energies[test_idx]
        ref_energies = full_ref_energies[test_idx]
        lemcmat = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)
        element_list = lmat.argmax(axis=1)
        xvec = get_xvec(weights, element_list)
        
        predict = lemcmat * xvec + ref_energies
        predict_loss = np.sqrt(np.mean((predict-exp_energies) ** 2))
        np.save("predict_"+str(n_cv),np.array(exp_energies))
        np.save("error_"+str(n_cv),np.array(predict-exp_energies))
        test_loss = [predict_loss]
        test_error.append(predict-exp_energies)
        print(f"CV #{n_cv}: Testing RMSE over {np.sum(~mask)} samples: {predict_loss:.3f}eV")

    print(f"Average {num_cross_val}-fold CV Training RMSE: {np.mean(train_loss)}eV")
    print(f"Average {num_cross_val}-fold CV Testing RMSE: {np.mean(test_loss)}eV")
    '''
    CV #0: Training RMSE over 4543 samples: 13.892eV
    CV #0: Testing RMSE over 649 samples: 13.480eV
    CV #1: Training RMSE over 4543 samples: 13.834eV
    CV #1: Testing RMSE over 649 samples: 13.920eV
    CV #2: Training RMSE over 4543 samples: 13.833eV
    CV #2: Testing RMSE over 649 samples: 16.225eV
    CV #3: Training RMSE over 4543 samples: 13.857eV
    CV #3: Testing RMSE over 649 samples: 14.542eV
    CV #4: Training RMSE over 4543 samples: 13.803eV
    CV #4: Testing RMSE over 649 samples: 14.124eV
    CV #5: Training RMSE over 4543 samples: 13.841eV
    CV #5: Testing RMSE over 649 samples: 14.316eV
    CV #6: Training RMSE over 4543 samples: 13.826eV
    CV #6: Testing RMSE over 649 samples: 13.952eV
    CV #7: Training RMSE over 4543 samples: 13.790eV
    CV #7: Testing RMSE over 649 samples: 14.243eV
    Average 8-fold CV Training RMSE: 13.834517864871225eV
    Average 8-fold CV Testing RMSE: 14.24309927757225eV
    '''

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
    Error Statistics:
    MAE: 11.4768
    STDEV: 10.3479
    RMSE: 14.3502
    MSE: 9.8850
    Max Error: 159.2767
    '''
