import scipy.constants as con

AU2CM = con.au * 1e2
KM2CM = 1E5
MSOL = 1.98847E30  # kg

NZ = {"H": (1, 0), "He": (2, 2), "Li": (3, 4), "Be": (4, 5),
      "B": (5, 6), "C": (6, 6), "N": (7, 7), "O": (8, 8),
      "F": (9, 10), "Ne": (10, 10), "Na": (11, 12),
      "Mg": (12, 12)}

# From Reynolds (1986)
a_k = 0.212
a_j = 6.5E-38
