import itertools
import sys
import json
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from src.smiletovectors import get_full_neighbor_vectors
from src.paulingelectro import get_eleneg_diff_mat
from rdkit import Chem

EN_MAT = get_eleneg_diff_mat()
au2eV = 27.21139

def complete_graph(graph):
    all_pairs = itertools.combinations(graph.nodes, 2)
    # Add edges between all pairs of nodes that are not already connected
    for u, v in all_pairs:
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, bond_type= "NONE")
    if len(graph.edges)==0:
        graph.add_edge(0,0, bond_type="NONE") # add node-node edge to give edge/edge attribute for single atom graphs
    return graph

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
		graph = nx.node_link_graph(string_dict[key], link="edges")
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

### Custom functions for processing the given data into pytorch-friendly format
def process_edge(graph):
    # add edges for all non linked nodes
    graph = complete_graph(graph)
    if len(graph.edges) > 0:
        u, v, edge_data = zip(*graph.edges(data=True))
        edge_index = torch.tensor(np.array([u, v]), dtype=torch.long)

        # Reformat edge data into numeric bond order
        # TODO: Add any additional edge features here
        def process_edge_data(edge_data):
            processed_data = np.ones((len(edge_data),1))
            for i, edge in enumerate(edge_data):
                if edge['bond_type'] == 'DOUBLE':
                    processed_data[i] = 2
                elif edge['bond_type'] == 'TRIPLE':
                    processed_data[i] = 3
                elif edge['bond_type'] == 'NONE':
                    processed_data[i] = 0
            return processed_data
        edge_attr = torch.tensor(process_edge_data(edge_data), dtype=torch.float32)
    else:
        edge_index = None
        edge_attr = None
    return edge_index, edge_attr

def process_nodes(node_raw_data, vectors):
    node_data = []
    dup_orbs = {}
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
         
        lmat = np.zeros((1,EN_MAT.shape[0]))
        lmat[0, Chem.Atom(symbol).GetAtomicNum() - 1] = 1
        
        e_neg_score = np.einsum('ij,jk,ki->i', lmat, EN_MAT, cmat)[0]        
        
        orbitals = n['orbitals']
        for i, orb in enumerate(orbitals):
            if type(orb) == int and orb == -1:
                orbitals[i] = None
        
        binding_energies = n['binding_energies']
        for i, energy in enumerate(binding_energies):
            if energy == -1:
                binding_energies[i] = None
                
        symbol = n['atom_type']
        formal_charge = n['formal_charge']
        
        # Remove empty entries
        orbitals = np.array(orbitals)
        orbitals = np.atleast_1d(orbitals[orbitals !=  None])
        if len(orbitals) == 0:
            continue
        
        binding_energies = np.array(binding_energies)
        binding_energies = binding_energies[binding_energies != None]
        if len(binding_energies) == 0:
            continue
        
        # If there are duplicate orbitals
        if len(orbitals) > 1:            
            dup_orbs[node_idx] = []
            for orb, exp_e in zip(orbitals, binding_energies):
                ref_e = -giveorbitalenergy(symbol, orb)
                n, l = get_n_l(orb)
                dup_orbs[node_idx] += [np.array([Chem.Atom(symbol).GetAtomicNum(), formal_charge, e_neg_score, n, l, ref_e, exp_e])]

        n, l = get_n_l(orbitals[0])
        ref_e = -giveorbitalenergy(symbol, orbitals[0])
        node_data += [np.array([Chem.Atom(symbol).GetAtomicNum(), formal_charge, e_neg_score, n, l, ref_e, binding_energies[0]])]
    node_data = np.atleast_2d(node_data)
    return node_data, dup_orbs

def networkx2torch(data):
    smiles = list(data.keys())
    # For each networkx graph, process the nodes and edges into torch_geometric.data.Data graphs
    torch_graphs = []
    for graph_num, smile in enumerate(smiles):
        graph = data[smile]
        print("System: ", smile)

        try:
            vectors = get_full_neighbor_vectors(smile)
        except:
            print(f"Skipping smile {smile} due to RDKit error")
            continue

        print(f"Processing graph {graph_num+1}/{len(smiles)}")
        
        ### Process Edge information  
        edge_index, edge_attr = process_edge(graph)

        ### Process Node information
        nodes, node_raw_data = zip(*graph.nodes(data=True))
        node_data, dup_orbs = process_nodes(node_raw_data, vectors)

        if len(node_data) == 0:
            continue
        
        # If any node has multiple orbital/binding_energy pairs
        if len(list(dup_orbs.keys())) > 0:
            # For each possible combination of nodes with variable orbital/binding_energies
            dups = list(itertools.product(*dup_orbs.values()))
            print(f"Found {len(dups)} duplicates in graph {graph_num}")
            for c in dups:
                # Update the node with a specific orbital/binding_energy
                for i, node_idx in enumerate(dup_orbs.keys()):
                    node_data[node_idx, :] = c[i].copy()
                # and save the information as it's own graph
                x1 = node_data[:,:-1].copy()
                y1 = node_data[:,-1].copy()
                torch_graph = Data(x1, edge_index, edge_attr, y1)
                torch_graph.smile = smile
                torch_graphs = torch_graphs + [torch_graph]
        else: # Or if there's just one possibility
            x = node_data[:,:-1]
            y = node_data[:,-1]
            torch_graph = Data(x, edge_index, edge_attr, y)
            torch_graph.smile = smile
            torch_graphs += [torch_graph]
    return torch_graphs

if __name__=='__main__':
    # Input arguments
    in_filename = sys.argv[1] # networkx .json file
    out_filename = 'graphs.pt' # File to save pyTorch graphs
    if len(sys.argv) > 2:
        out_filename = sys.argv[2]
    
    # Load in networkx graphs
    data = load_data_from_file(in_filename)
    torch_graphs = networkx2torch(data)
    print(f"Saving a total of {len(torch_graphs)} PyTorch Graphs")
    with open(out_filename, 'wb') as f:
        torch.save(torch_graphs, f)
