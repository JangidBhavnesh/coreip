# Let's create the model-0
# This is the look up table for calculating the orbital energies based
# on the atomic number and effective nuclear charges
import json
import matplotlib.pyplot as plt
import numpy as np

element_symbols = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "I", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

# Accoardin to the Slater's rules the effective nuclear charge is given by
# Z_eff = Z - S

# So, the orbital energies are given by
# E = -13.64 * Z_eff^2 / n^2

def _get_atomic_number(symbol):
    """
    Get the atomic number of an element given its symbol.
    """
    try:
        return element_symbols.index(symbol) + 1
    except ValueError:
        raise ValueError(f"Element symbol '{symbol}' not found in the list.")
    
def get_l(l):
    '''
    l = s, p, d, f, g
    '''
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g':4,}.get(l, "l value is not valid")

def get_n_l(orb):
    n = int(orb[0])
    l = get_l(str(orb[1]))
    return n, l

def max_electrons_per_l(l):
    return 2*(2*l + 1)


 # Electron configuration order up to Z=36 (Krypton)
aufbauorder = [
    (1, 0), # 1s
    (2, 0), (2, 1), # 2s, 2p
    (3, 0), (3, 1), # 3s, 3p
    (4, 0), (3, 2), (4, 1), # 4s, 3d, 4p
    (5, 0), (4, 2), (5, 1), # 5s, 4d, 5p
    (6, 0), (4, 3), (5, 2), (6, 1), # 6s, 4f, 5d, 6p
]
    
def slater_zeff(z, n_given, l_given):
    '''
    z: atomic number
    n_given: principle quantum number
    l_given: azimuthal quantum number
    '''
    electrons = []
    remaining = z

    for (n, l) in aufbauorder:
        capacity = 2*(2*l+1)
        if remaining >= capacity:
            electrons.append((n, l, capacity))
            remaining -= capacity
        else:
            electrons.append((n, l, remaining))
            break

    S = 0
    for (n, l, count) in electrons:
        if (n, l) == (n_given, l_given):
            count -= 1  # exclude the target electron
            S += 0.35 * count
        elif n == n_given - 1:
            if l_given <= 1: # s or p electron
                S += 0.85 * count
            else: # d or f electron
                S += 1.00 * count
        elif n < n_given - 1:
            S += 1.00 * count
        elif n == n_given and l != l_given:
            S += 0.35 * count  # same principal shell but different subshell

    Zeff = z - S
    return Zeff

def _get_orbital_energy(symbol, orb):
    z = _get_atomic_number(symbol)
    n, l = get_n_l(orb)
    zeff = slater_zeff(z, n, l)
    assert zeff <= z
    orbenergy = -13.64*zeff*zeff/(n*n)
    return orbenergy

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
                cal_energy = -1. * _get_orbital_energy(atom, orbital_prefix) # Koopman's theorem, IP = -HOMO
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
