import itertools
import sys
import json
import pandas as pd
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

def giveorbitalenergy(ele, orb, orbital_energy_file='./src/orbitalenergy.json'):
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
            processed_data = np.zeros((len(edge_data),7))
            for i, edge in enumerate(edge_data):
                if edge['bond_type'] == 'DOUBLE':
                    processed_data[i][2] = 1
                elif edge['bond_type'] == 'TRIPLE':
                    processed_data[i][3] = 1
                elif edge['bond_type'] == 'QUADRUPLE':
                    processed_data[i][4] = 1
                elif edge['bond_type'] == 'NONE':
                    processed_data[i][0] = 1
                elif edge['bond_type'] == 'AROMATIC':
                    processed_data[i][5] = 1
                elif edge['bond_type'] == 'DATIVE':
                    processed_data[i][6] = 1
            return processed_data
        edge_attr = torch.tensor(process_edge_data(edge_data), dtype=torch.float32)
    else:
        edge_index = None
        edge_attr = None
    return edge_index, edge_attr

def process_nodes(node_raw_data, vectors, smile):
    node_data = []
    dup_orbs = {}
    temp_list = []
    df = pd.DataFrame(columns=["Molecule", "Element", "Orbital", "Count"])
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
        atoms = n['atom_type']
        binding_energies = n['binding_energies']
        for i, orb in enumerate(orbitals):
            df_tmp = pd.DataFrame({"Molecule": [smile], "Element": [atoms], "Orbital": [orbitals[i]], "Count": [i]})
            df = pd.concat([df, df_tmp], ignore_index=False)
            if type(orb) == int and orb == -1:
                orbitals[i] = -1#None
        
        for i, energy in enumerate(binding_energies):
            if energy == -1:
                binding_energies[i] = -1#None
                
        symbol = n['atom_type']
        formal_charge = n['formal_charge']
        
        # Remove empty entries
        orbitals = np.array(orbitals)
        
        orbitals = np.atleast_1d(orbitals[orbitals !=  None])
        binding_energies = np.array(binding_energies)
        binding_energies = binding_energies[binding_energies != None]
        if len(orbitals) == 0:
            return None, None, None
        
        if len(binding_energies) == 0:
            return None, None, None
        
        # If there are duplicate orbitals
        if len(orbitals) > 1:            
            dup_orbs[node_idx] = []
            for orb, exp_e in zip(orbitals, binding_energies):
                if exp_e == -1:
                    n=-1
                    l=-1
                    ref_e = -1
                else:
                    ref_e = -giveorbitalenergy(symbol, orb)
                    n, l = get_n_l(orb)
                dup_orbs[node_idx] += [np.array([Chem.Atom(symbol).GetAtomicNum(), formal_charge, e_neg_score, n, l, ref_e, exp_e])]
        if orb == '-1':
            n = -1
            l = -1
            ref_e = -1
        else:
            n, l = get_n_l(orbitals[0])
            ref_e = -giveorbitalenergy(symbol, orbitals[0])
        node_data += [np.array([Chem.Atom(symbol).GetAtomicNum(), formal_charge, e_neg_score, n, l, ref_e, binding_energies[0]])]

    node_data = np.atleast_2d(node_data)
    return node_data, dup_orbs, df


def networkx2torch(data):
    smiles = list(data.keys())
    # For each networkx graph, process the nodes and edges into torch_geometric.data.Data graphs
    torch_graphs = []
    df = pd.DataFrame(columns=["Molecule", "Element", "Orbital"])
    for graph_num, smile in enumerate(smiles):
        graph = data[smile]
        try:
            # print(smile)
            vectors = get_full_neighbor_vectors(smile)
        except:
            print(f"Skipping smile {smile} due to RDKit error")
            continue

        # print(f"Processing graph {graph_num+1}/{len(smiles)}")
        
        ### Process Edge information  
        edge_index, edge_attr = process_edge(graph)

        ### Process Node information
        nodes, node_raw_data = zip(*graph.nodes(data=True))
        node_data, dup_orbs, df_tmp = process_nodes(node_raw_data, vectors, smile)
        if df_tmp["Count"].max()>0:
            df_sorted = df_tmp.sort_values(by="Count")
        else:
            df_sorted = df_tmp
        df = pd.concat([df, df_sorted[["Molecule", "Element", "Orbital"]]], ignore_index=False)
        if node_data is None or dup_orbs is None:
            continue
        
        # If any node has multiple orbital/binding_energy pairs
        if len(list(dup_orbs.keys())) > 0:
            lis_int = [*dup_orbs.values()]
            all_dups = []
            for i in range(len(lis_int[0])): # for each value in node 1
                this_dup = []
                for k in range(len(lis_int)): # loop over nodes
                    this_dup = this_dup + [np.array(lis_int[k][i])]
                all_dups = all_dups+[this_dup]

            dups = all_dups
            # print(f"Found {len(dups)} duplicates in graph {graph_num}")
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
    
    df.to_csv("DF_Formatted.csv", index=False)
    return torch_graphs

if __name__=='__main__':
    # Input arguments
    in_filename = sys.argv[1] # networkx .json file
    out_filename = 'graphs.pt' # File to save pyTorch graphs
    if len(sys.argv) > 2:
        out_filename = sys.argv[2]
    
    # Load in networkx graphs
    data = load_data_from_file(in_filename)
    data = data
    torch_graphs = networkx2torch(data)
    print(f"Saving a total of {len(torch_graphs)} PyTorch Graphs")
    num_ys = 0
    for d in torch_graphs:
        num_ys+=len(d.y)
    print("Total number of targets: ", num_ys)
    with open(out_filename, 'wb') as f:
        torch.save(torch_graphs, f)
