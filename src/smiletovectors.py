
from rdkit import Chem
import numpy as np

def get_full_neighbor_vectors(smiles):
    '''
    This function will generate a vector of size (100, 1) 
    for all non-H atoms in given smile notation. Where each index 
    number denotes the number of that atoms in the surrounding of 
    the current atom.
    For example: CH4:
    C: [4, 0, 0, ...0]
    For CO
    C: [0, 0, 0, 0, 0, 0, 0, 1, 0, ...0]
    O: [0, 0, 0, 0, 0, 1, 0, 0, ....0]
    '''
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    atom_vectors = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == 'H':
            continue 

        vec = np.zeros((100, 1), dtype=int)

        for nbr in atom.GetNeighbors():
            atomic_num = nbr.GetAtomicNum()
            if atomic_num < 100:
                vec[atomic_num-1][0] += 1

        atom_vectors.append((atom.GetIdx(), atom.GetSymbol(), vec))

    return atom_vectors

if __name__=='__main__':
    smiles = "CCO" # #CH3CH2OH
    vectors = get_full_neighbor_vectors(smiles)

    for idx, symbol, vec in vectors:
        print(f"Atom {idx} ({symbol})")
        print(vec.T)
