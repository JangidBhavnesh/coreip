

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
smilelst = _get_all_smiles(data)

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

valay_old_smile = smilelst[:350]
bhavnesh_old_smile = smilelst[350:500]
jacob_old_smile = smilelst[500:650]
joshua_old_smile = smilelst[650:]

valay_new_smile = _give_smilelst('valay.json')
bhavnesh_new_smile = _give_smilelst('bhavnesh.json')
jacob_new_smile = _give_smilelst('jacob.json')
joshua_new_smile = _give_smilelst('joshua.json')


filelst = ['valay.json', 'bhavnesh.json', 'jacob.json', 'joshua.json']
for file in filelst:
    print(file)
    Invsmile = check_updated_json(file)
    if len(Invsmile) > 0:
        print("For file", file, "these strings are problematic", Invsmile)

