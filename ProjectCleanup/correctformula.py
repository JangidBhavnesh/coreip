

import json
import sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def _get_all_smiles(data):
   smiltslst = [key for key in data.keys()]
   return smiltslst

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

# Total number of smiles in the data
smilelst = _get_all_smiles(data)

for smiles in smilelst:
    mol = Chem.MolFromSmiles(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    ntime = len(formula)
    data[smiles]["chemical_formula"] = [formula,]*ntime
    
    with open(sys.argv[1], 'w') as f:
        json.dump(data, f, indent=4)
