import itertools
import sys
import json
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data

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

### Custom functions for processing the given data into pytorch-friendly format
def process_edge(graph):
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
            return processed_data
        edge_attr = torch.tensor(process_edge_data(edge_data), dtype=torch.float16)
    else:
        edge_index = None
        edge_attr = None
    return edge_index, edge_attr

def process_nodes(node_raw_data):
    node_data = []
    dup_orbs = {}
    for node_idx, data in enumerate(node_raw_data):
        orbitals = data['orbitals']
        for i, orb in enumerate(orbitals):
            if type(orb) == int and orb == -1:
                orbitals[i] = np.NaN
        
        binding_energies = data['binding_energies']
        for i, energy in enumerate(binding_energies):
            if energy == -1:
                binding_energies[i] = np.NaN
                
        atom_type = data['atom_type']
        formal_charge = data['formal_charge']
        
        # If there are duplicate orbitals
        if len(orbitals) > 1:
            orbitals = np.array(orbitals)
            orbitals = orbitals[orbitals != 'nan']
            if len(orbitals) == 0:
                orbitals = ['nan']
            
            binding_energies = np.array(binding_energies)
            binding_energies = binding_energies[~np.isnan(binding_energies)]
            if len(binding_energies) == 0:
                binding_energies = [np.NaN]
            
            dup_orbs[node_idx] = []
            for orb, energy in zip(orbitals, binding_energies):
                dup_orbs[node_idx] += [[atom_type, formal_charge, orb, energy]]
        node_data += [[atom_type, formal_charge, orbitals[0], binding_energies[0]]]
    node_data = np.atleast_2d(node_data)
    return node_data, dup_orbs

def networkx2torch(data):
    smiles = list(data.keys())
    netx_graphs = list(data.values())
    
    # For each networkx graph, process the nodes and edges into torch_geometric.data.Data graphs
    torch_graphs = []
    for graph_num, temp in enumerate(zip(smiles, netx_graphs)):
        smile_str, graph = temp
        print(f"Processing graph {graph_num+1}/{len(netx_graphs)}")
        ### Process Edge information  
        edge_index, edge_attr = process_edge(graph)

        ### Process Node information
        nodes, node_raw_data = zip(*graph.nodes(data=True))
        node_data, dup_orbs = process_nodes(node_raw_data)

        # If any node has multiple orbital/binding_energy pairs
        if len(list(dup_orbs.keys())) > 0:
            # For each possible combination of nodes with variable orbital/binding_energies
            dups = list(itertools.product(*dup_orbs.values()))
            print(f"Found {len(dups)} duplicates in graph {graph_num}")
            for c in dups:
                # Update the node with a specific orbital/binding_energy
                for i, node_idx in enumerate(dup_orbs.keys()):
                    node_data[node_idx, :] = c[i]
                # and save the information as it's own graph
                x = node_data[:,:-1]
                y = node_data[:,-1]
                torch_graph = Data(x, edge_index, edge_attr, y)
                torch_graph.smile = smile_str
                torch_graphs += [torch_graph]
        else: # Or if there's just one possibility
            x = node_data[:,:-1]
            y = node_data[:,-1]
            torch_graph = Data(x, edge_index, edge_attr, y)
            torch_graph.smile = smile_str
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
