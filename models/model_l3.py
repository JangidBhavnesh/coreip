import itertools
import sys
import json
from matplotlib.font_manager import weight_dict
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from src.smiletovectors import get_full_neighbor_vectors
from src.paulingelectro import get_eleneg_diff_mat
import matplotlib.pyplot as plt

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

from rdkit import Chem
if __name__=='__main__':
    # Input arguments
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
        except:
            print(f"Skipping smile {smile} due to RDKit error")
            continue
            
        nodes, node_raw_data = zip(*data[smile].nodes(data=True))
        
        temp_list = []
        for n in node_raw_data:
            if n['atom_type'] == 'H':
                continue
            temp_list += [n]
        node_raw_data = tuple(temp_list)

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
                # if np.abs(orb_en - binding_e) > 40:
                #     print(smile, orb)
                ref_energies += [orb_en]
    full_lmat = np.array(lmat)
    full_lmat = np.atleast_2d(full_lmat)
    full_cmat = np.array(cmat).squeeze().T
    full_cmat = full_cmat.reshape(full_lmat.T.shape)
    full_exp_energies = np.array(exp_energies)
    full_ref_energies = np.array(ref_energies)
    
    null_loss = np.sqrt(np.mean((full_ref_energies - full_exp_energies)**2))
    print(f"Null Loss: {null_loss:.3f}eV")
    plt.scatter(full_exp_energies, full_ref_energies,label=f'RMSE={null_loss:.3f}eV')
    plt.xlabel('Experimental Energy (eV)')
    plt.ylabel('Reference Orbital Energies (eV)')
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot([-100, 10000], [-100, 10000], c='k')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title("Null Model: Compare to reference orbital energy")
    plt.legend()
    plt.savefig("model_null.png")

    train_samples = 2500
    test_samples = len(full_lmat) - train_samples
    
    indices = np.arange(len(full_lmat))
    np.random.shuffle(indices)
    
    train_idx = indices[:train_samples]
    lmat = full_lmat[train_idx,:]
    cmat = full_cmat[:, train_idx]
    exp_energies = full_exp_energies[train_idx]
    ref_energies = full_ref_energies[train_idx]
    
    lemcmat = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)
    lemcmat = np.vstack([lemcmat, np.ones(lemcmat.shape)]).T
    
    element_list = lmat.argmax(axis=1)
    weights = np.zeros((cmat.shape[0]*2))
    
    def get_xvec(weights, element_list):
        return np.array([weights[el] for el in element_list])

    def errorfunc(x):
        xvec = get_xvec(x.reshape(cmat.shape[0],2), element_list)
        loss = np.sqrt(np.mean((np.sum(lemcmat * xvec, axis=1) + ref_energies - exp_energies) ** 2))
        return loss
    
    from scipy import optimize
    np.set_printoptions(threshold=1000)
    # print(ref_energies - exp_energies)
    results = optimize.minimize(errorfunc, weights)
    weights = results['x']
    # print(f'Weights: {weights}')
    print(f"Training RMSE over {train_samples} samples: {results['fun']:.3f}eV")
    
    lmat = full_lmat[train_samples:train_samples+test_samples,:]
    cmat = full_cmat[:, train_samples:train_samples+test_samples]
    exp_energies = full_exp_energies[train_samples:train_samples+test_samples]
    ref_energies = full_ref_energies[train_samples:train_samples+test_samples]
    
    lemcmat = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)
    lemcmat = np.vstack([lemcmat, np.ones(lemcmat.shape)]).T
    element_list = lmat.argmax(axis=1)
    
    xvec = get_xvec(weights.reshape(cmat.shape[0],2), element_list)
    print(lemcmat.shape, xvec.shape, cmat.shape, exp_energies.shape, ref_energies.shape)
    predict = np.sum(lemcmat * xvec, axis=1) + ref_energies
    predict_loss = np.sqrt(np.mean((predict-exp_energies) ** 2))
    print(f"Testing RMSE over {test_samples} samples: {predict_loss:.3f}eV")
   
    plt.figure()
    plt.scatter(exp_energies, predict, label=f'RMSE={predict_loss:.3f}eV')
    plt.xlabel('Experimental Energy (eV)')
    plt.ylabel('Predicted Energies (eV)')
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot([-100, 100000], [-100, 100000], c='k')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title("Model V0: Linear fit to Electronegativity Environment")
    plt.legend()
    plt.savefig("model_v0.png")
