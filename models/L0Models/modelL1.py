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
    print(f"{ele}: {orb} orbital energy is {cbenergy} eV")
    return cbenergy

print(giveorbitalenergy('Ag', '3d'))

