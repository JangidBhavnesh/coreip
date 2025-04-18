import sys
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
    # Input arguments
    in_filename = sys.argv[1] # networkx .json file
        
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
        
        lmat = full_lmat[train_idx, :]
        cmat = full_cmat[:, train_idx]
        exp_energies = full_exp_energies[train_idx]
        ref_energies = full_ref_energies[train_idx]
        lemcmat = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)
        lemcmat = np.vstack([lemcmat, np.ones(lemcmat.shape)]).T
        element_list = lmat.argmax(axis=1)
        weights = np.zeros(cmat.shape[0]*2)

        def errorfunc(x):
            xvec = get_xvec(x.reshape(cmat.shape[0],2), element_list)
            loss = np.sqrt(np.mean((np.sum(lemcmat * xvec,axis=1) + ref_energies - exp_energies) ** 2))
            return loss
    
        results = optimize.minimize(errorfunc, weights)
        weights = results['x']
        with open(f"model_l3_CV{n_cv}_weights.txt", 'w') as f:
            f.write(str(weights.reshape(cmat.shape[0],2)))
            
        train_loss += [results['fun']]
        print(f"CV #{n_cv}: Training RMSE over {np.sum(mask)} samples: {results['fun']:.3f}eV")

        lmat = full_lmat[test_idx, :]
        cmat = full_cmat[:, test_idx]
        exp_energies = full_exp_energies[test_idx]
        ref_energies = full_ref_energies[test_idx]
        lemcmat = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)
        lemcmat = np.vstack([lemcmat, np.ones(lemcmat.shape)]).T
        element_list = lmat.argmax(axis=1)
        xvec = get_xvec(weights.reshape(cmat.shape[0],2), element_list)
        
        predict = np.sum(lemcmat * xvec, axis=1) + ref_energies
        predict_loss = np.sqrt(np.mean((predict-exp_energies) ** 2))
        test_loss = [predict_loss]
        print(f"CV #{n_cv}: Testing RMSE over {np.sum(~mask)} samples: {predict_loss:.3f}eV")

    print(f"Average {num_cross_val}-fold CV Training RMSE: {np.mean(train_loss)}eV")
    print(f"Average {num_cross_val}-fold CV Testing RMSE: {np.mean(test_loss)}eV")
    
