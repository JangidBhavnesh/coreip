import numpy as np

# Note: I have replace the He and Ar electronegativity with 0.

pdata='''
1       H       hydrogen        2.20
2       He      helium          0  
3       Li      lithium         0.98
4       Be      beryllium       1.57
5       B       boron           2.04
6       C       carbon          2.55
7       N       nitrogen        3.04
8       O       oxygen          3.44
9       F       fluorine        3.98
10      Ne      neon            0
11      Na      sodium          0.93
12      Mg      magnesium       1.31
13      Al      aluminium       1.61
14      Si      silicon         1.90
15      P       phosphorus      2.19
16      S       sulfur          2.58
17      Cl      chlorine        3.16
18      Ar      argon           0
19      K       potassium       0.82
20      Ca      calcium         1.00
21      Sc      scandium        1.36
22      Ti      titanium        1.54
23      V       vanadium        1.63
24      Cr      chromium        1.66
25      Mn      manganese       1.55
26      Fe      iron            1.83
27      Co      cobalt          1.88
28      Ni      nickel          1.91
29      Cu      copper          1.90
30      Zn      zinc            1.65
31      Ga      gallium         1.81
32      Ge      germanium       2.01
33      As      arsenic         2.18
34      Se      selenium        2.55
35      Br      bromine         2.96
36      Kr      krypton         3.00
37      Rb      rubidium        0.82
38      Sr      strontium       0.95
39      Y       yttrium         1.22
40      Zr      zirconium       1.33
41      Nb      niobium         1.6
42      Mo      molybdenum      2.16
43      Tc      technetium      1.9
44      Ru      ruthenium       2.2
45      Rh      rhodium         2.28
46      Pd      palladium       2.20
47      Ag      silver          1.93
48      Cd      cadmium         1.69
49      In      indium          1.78
50      Sn      tin             1.96
51      Sb      antimony        2.05
52      Te      tellurium       2.1
53      I       iodine          2.66
54      Xe      xenon           2.6
55      Cs      caesium         0.79
56      Ba      barium          0.89
57      La      lanthanum       1.10
58      Ce      cerium          1.12
59      Pr      praseodymium    1.13
60      Nd      neodymium       1.14
61      Pm      promethium      1.15
62      Sm      samarium        1.17
63      Eu      europium        1.15
64      Gd      gadolinium      1.20
65      Tb      terbium         1.1
66      Dy      dysprosium      1.22
67      Ho      holmium         1.23
68      Er      erbium          1.24
69      Tm      thulium         1.25
70      Yb      ytterbium       1.15
71      Lu      lutetium        1.27
72      Hf      hafnium         1.3
73      Ta      tantalum        1.5
74      W       tungsten        2.36
75      Re      rhenium         1.9
76      Os      osmium          2.2
77      Ir      iridium         2.20
78      Pt      platinum        2.28
79      Au      gold            2.54
80      Hg      mercury         2.00
81      Tl      thallium        1.62
82      Pb      lead            2.33
83      Bi      bismuth         2.02
84      Po      polonium        2.0
85      At      astatine        2.2
86      Rn      radon           2.2
87      Fr      francium        0.78
88      Ra      radium          0.9
89      Ac      actinium        1.1
90      Th      thorium         1.3
91      Pa      protactinium    1.5
92      U       uranium         1.38
93      Np      neptunium       1.36
94      Pu      plutonium       1.28
95      Am      americium       1.3
96      Cm      curium          1.3
97      Bk      berkelium       1.3
98      Cf      californium     1.3
99      Es      einsteinium     1.3
100     Fm      fermium         1.3
101     Md      mendelevium     1.3
102     No      nobelium        1.3
'''

def give_paulingdata(pdata):
    paulingdata = {}

    for line in pdata.strip().splitlines():
        tempdata = line.strip().split()
        atomnum = int(tempdata[0])
        electronegativity = tempdata[-1]
        try:
            electronegativity = float(electronegativity)
        except ValueError:
            electronegativity = np.nan
        paulingdata[atomnum] = electronegativity
    return paulingdata

def get_eleneg_diff_mat(num_elements=100):
    # Creation of relative electronegativity matrix
    paulingdata = give_paulingdata(pdata)
    data = np.array(list(paulingdata.values()))
    elenegMat = data[:num_elements,None]-data[None, :num_elements]
    return elenegMat
    
# Next: 