# Let's create the model-1
# What are we trying to do it learn the delta of the core-ip
# As in the core-ip is shift in positive or negative from reference which will give info of 
# the enviroment.

# Generate the electronegativity table:
paulingdata = {}


import numpy as np
import scipy

natom = 140
nele = 100
lmat = np.zeros([natom, nele])
lmat[0:50, 6] = 1
lmat[50:90, 7] = 1
lmat[90:110, 8] = 1
lmat[110:140, 16] = 1

enmat = np.random.rand(100)
enmat = enmat[:, None] - enmat[None, :]

cmat = np.random.randint(2, size=(100, 140))

lemcmat = (lmat@enmat)@cmat
lemcmat = np.einsum('ii->i', lemcmat)

cexp = np.random.randint(280, 295, size=(50))
nexp = np.random.randint(390, 410, size=(40))
oexp = np.random.randint(530, 545, size=(20))
sexp = np.random.randint(800, 820, size=(30))
eexp = np.hstack([cexp, nexp, oexp, sexp]).reshape(-1,1)

cref = [290, ]*50
nref = [400, ]*40
oref = [540, ]*20
sref = [810, ]*30
eref = np.hstack([cref, nref, oref, sref]).reshape(-1,1)

def getmexvec(x, freq):
  xvec = np.array([xi for xi, freqi in zip(x, freq) for _ in range(freqi) ])
  # xvec = np.array([[xi for _ in range(freqi)] for xi, freqi in zip(x, freq)])
  return xvec

# lencmat, xvec + eref - eexp = 0
freq = [50, 40, 20, 30]

def errorfunc(x):
  xvec = getmexvec(x, freq)
  loss = np.linalg.norm(lemcmat * xvec + eref - eexp)
  return loss
from scipy import optimize
x0 = np.ones(4)
optimize.minimize(errorfunc, x0, method='nelder-mead')


import json

au2eV = 27.21139

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

    with open('../../src/orbitalenergy.json', 'r') as f:
        data = json.load(f)
    try:
        orbenegele = data[ele]
        del data
    except KeyError:
        raise KeyError("Element symbol not found")
    
    n, l = get_n_l(orb)
    cbenergy = orbenegele[str(l)][n-l-1]
    cbenergy *= au2eV
    #print(f"{ele}: {orb} orbital energy is {cbenergy} eV")
    return cbenergy

def _get_error(data):
    errors = []
    for tempdata in data.values():
        for node in tempdata['nodes']:
            refbe = node['binding_energies']
            atom = node['atom_type']
            orbitals = node['orbitals']

            assert len(orbitals) == len(refbe), 'Something is wrong with this data'

            for orb, ref_energy in zip(orbitals, refbe):
                if orb == -1:
                    break
                orbital_prefix = orb[:2]
                cal_energy = -1. * giveorbitalenergy(atom, orbital_prefix) # Koopman's theorem, IP = -HOMO
                errors.append(cal_energy - ref_energy)
    return errors



if __name__ == '__main__':

    with open('graph_data.json', 'r') as f:
        data = json.load(f)
    
    errors = np.array(_get_error(data))
    
    # Error Statistics
    mae = np.mean(np.abs(errors))
    stdev = np.std(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(np.abs(errors))

    stats = {
        'MAE': mae,
        'STDEV': stdev,
        'RMSE': rmse,
        'MSE': mean_error, # mean-signed error
        'Max Error': max_error}

    print("\nError Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # # Plotting the erros
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.bar(stats.keys(), stats.values())
    # ax.set_ylabel('Error Value')
    # ax.set_title('Error Statistics')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()



