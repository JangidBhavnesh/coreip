

import json
import sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def _get_all_smiles(data):
   smiltslst = [key for key in data.keys()]
   return smiltslst

with open('raw_data.json', 'r') as f:
    data = json.load(f)

# Total number of smiles in the data
smilelstorg = _get_all_smiles(data)

def _get_rdkit_filter(smilelst):
    invalid_smiles = []
    for smi in smilelst:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if mol is None:
                invalid_smiles.append(smi)
        except Exception as e:
            invalid_smiles.append(smi)
    return invalid_smiles

def _get_rdkit_filter_smilelst(smilelst):
    invalidsmiles = _get_rdkit_filter(smilelst)
    smilelst = [s for s in smilelst if s not in invalidsmiles]
    return smilelst

def _get_more_nonH_atoms(smilelst):
    smilelst = _get_rdkit_filter_smilelst(smilelst)
    smiletotakecare = []
    for smiles in smilelst:
        mol = Chem.MolFromSmiles(smiles)
        atom_counts = {}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'H':
                atom_counts[atom.GetSymbol()] = atom_counts.get(atom.GetSymbol(), 0) + 1
        for atom, count in atom_counts.items():
            if count > 1:
                smiletotakecare.append(smiles)
                break
    return smiletotakecare

def _give_smilelst(jsonfile):
    with open(jsonfile, 'r') as f:
        tempdata = json.load(f)
        smilelst = _get_all_smiles(tempdata)
        del tempdata
    return smilelst

def check_updated_json(file):
    smilelst = _give_smilelst(file)
    with open(file, 'r') as f:
        temdata = json.load(f)
    Invalidcorrections=[]
    for smiles in smilelst:
        moldata = temdata[smiles]
        lengths = [len(value) for value in moldata.values()]
        if not all(length == lengths[0] for length in lengths):
            Invalidcorrections.append(smiles)
    return Invalidcorrections

def _find_added_smiles(smilelst1, smilelst2):
    newsmiles = [s1 for s1 in smilelst1 if s1 not in smilelst2]
    return newsmiles

# Filtered smile strings
smilelst = _get_more_nonH_atoms(smilelstorg)

valay_old_smile = smilelst[:350]
bhavnesh_old_smile = smilelst[350:500]
jacob_old_smile = smilelst[500:650]
joshua_old_smile = smilelst[650:]

valay_new_smile = _give_smilelst('valay.json')
bhavnesh_new_smile = _give_smilelst('bhavnesh.json')
jacob_new_smile = _give_smilelst('jacob.json')
joshua_new_smile = _give_smilelst('joshua.json')

filelst = ['valay.json', 'bhavnesh.json', 'jacob.json', 'joshua.json']
smillst1 = [valay_new_smile, bhavnesh_new_smile, jacob_new_smile, joshua_new_smile]
smillst2 = [valay_old_smile, bhavnesh_old_smile, jacob_old_smile, joshua_old_smile]

for i, file in enumerate(filelst):
    print(file)
    newaddition = _find_added_smiles(smillst1[i], smillst2[i])
    print("Total new additions:", len(newaddition))
    if len(newaddition) > 0:
       print("New smiles:", newaddition)
    Invsmile = check_updated_json(file)
    if len(Invsmile) > 0:
        print("For file", file, "these strings are problematic", Invsmile)


newdata = {}
# Adding smiles from cleaned up json
for f in filelst:
    with open(f, 'r') as infile:
        tempdata = json.load(infile)
        newdata.update(tempdata)

# Adding smiles which were not changed at all.
newdata.update({smiles: data[smiles] for smiles in smilelstorg if smiles not in smilelst})

with open('coreip_raw_data.json', 'w') as outfile:
    json.dump(newdata, outfile, indent=2)