# Let's create the model-0
# This is the look up table for calculating the orbital energies based
# on the atomic number and effective nuclear charges

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

print(_get_orbital_energy('C', '1s'))
