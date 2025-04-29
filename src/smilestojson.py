import json
from rdkit import Chem
import numpy as np

smilelst ='''
CF
C(F)F
C(=O)(F)F
C(F)(F)F
C(N=O)(F)(F)F
C(F)(F)(F)F
C#N
C(#N)N
C=O
C(=O)O
C(=O)N
C
C(=O)(N)N
CO
CN
C(#N)N=O
C(=O)=O
C=CF
CCF
C=C(F)F
C(=C(F)F)F
CC(F)(F)F
C(=C(F)F)(F)F
C(C(F)(F)F)(F)(F)F
C(N(C(F)(F)F)[O])(F)(F)F
C(OC(F)(F)F)(F)(F)F
C(OOC(F)(F)F)(F)(F)F
C(OOOC(F)(F)F)(F)(F)F
C#C
C=C=O
CC#N
C=C
C(#N)N=C(N)N
CC=O
CC
CCO
COC
C(#N)C#N
C(=O)(C(F)(F)F)C(F)(F)F
C(C(F)(F)F)(C(F)(F)F)(F)F
CC#C
C=CC=O
CC=C
CC(=O)C
CC(=O)OC
CCC(=O)O
CN(C)C=O
CC(C)[N+](=O)[O-]
CCC
CN(C)C
C[N+](C)(C)[O-]
C(=C=O)=C=O
CCOC(=O)CF
CCOC(=O)C(F)F
CCOC(=O)C(F)(F)F
C(#CC(F)(F)F)C(F)(F)F
C(=C(/C(F)(F)F)\F)(\C(F)(F)F)/F
C1(C(C(C1(F)F)(F)F)(F)F)(F)F
C(C(F)(F)F)(C(F)(F)F)(C(F)(F)F)O
C1=COC=C1
C1=CNC=C1
CC#CC
CC(=O)OC=C
CC(=O)OC(=O)C
CCOC(=O)C
CCCC
CC(C)(C)C#N
CCC(=O)OCC
CC(C)(C)C
CCCCC
C1=CC(=CC=C1[N+](=O)[O-])F
C1=CC=C(C=C1)F
C1=CC(=CC(=C1)F)N
C1=CC=C(C(=C1)N)F
C1=C(C=C(C=C1F)F)F
C1=C(C(=C(C(=C1F)F)F)F)F
C1(=C(C(=C(C(=C1F)F)F)F)F)F
C1=CC=CC=C1
C1=CC=C(C=C1)O
C1=CC=C(C=C1)N
C1CCCCC1
CCCCCC
C(#N)C1(C(O1)(C#N)C#N)C#N
C1=CC(=CC=C1C#N)F
C1=CC=C(C=C1)C(F)(F)F
C1=CC=C(C=C1)C#N
[2H]C1=C(C(=C(C(=C1[2H])[2H])C#N)[2H])[2H]
C1=CC(=CC=C1C=O)[N+](=O)[O-]
C1=CC=C(C(=C1)C#N)N
C1=CC=C(C=C1)C=O
CC1=CC=CC=C1
C1=CC(=CC=C1C#N)C(F)(F)F
CC(=O)C1=CC=CC=C1
CN(C)C1=CC=C(C=C1)N
CCCCCCCC
CC(C)(C)N(C(C)(C)C)[O]
C1=CC=C(C=C1)/C=C/C=O
CC1=CC(=C(C=C1)C(=O)C)C
CC1=CC(=C(C=C1C)C(=O)C)C
C1=CC=C(C=C1)C(=O)C2=CC=CC=C2
C1=CC=C(C=C1)OC(=O)OC2=CC=CC=C2
CCCCCCCCCCCCC
'''

def checksmiles(smiles, add_bonds=True):
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

    If add_bonds is turned on, it multiplies the bonds with neighbors
    For example: Propene (H2C=CH-CH3)
    C:[2,0,0,0,0,2,....] first carbon
    C:[1,0,0,0,0,3,....] second carbon
    C:[3,0,0,0,0,1,....] third carbon
    '''
    try: 
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    except:
        print("No")
        mol = Chem.MolFromSmiles(smiles,sanitize=False) ## for bad cases, doesn't always get the right answer

    return True

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [],
        "edges": []
    }
    
    # Add nodes
    for atom in mol.GetAtoms():
        node = {
            "atom_type": atom.GetSymbol(),
            "formal_charge": atom.GetFormalCharge(),
            "orbitals":['1s'],
            "id": atom.GetIdx()
        }
        graph["nodes"].append(node)
    
    # Add edges
    bond_type_map = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC"
    }
    
    for bond in mol.GetBonds():
        edge = {
            "bond_type": bond_type_map.get(bond.GetBondType(), "UNKNOWN"),
            "source": bond.GetBeginAtomIdx(),
            "target": bond.GetEndAtomIdx()
        }
        graph["edges"].append(edge)
    
    return graph

def generate_graph_from_smile(smiles):
    return smiles_to_graph(smiles)

data = {}

for sm in smilelst.split():
    data[sm] = generate_graph_from_smile(sm)

with open("extrachallenege.json", 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=True)
