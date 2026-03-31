import numpy as np
from scipy.special import comb

def RandIndex(c1, c2):
    # Check if inputs are valid
    if len(c1) != len(c2) or len(c1) == 0 or len(c2) == 0:
        raise ValueError('RandIndex: Requires two non-empty vectors of equal length')

    # Form contingency matrix
    unique_c1 = np.unique(c1)
    unique_c2 = np.unique(c2)
    n = len(c1)
    C = np.zeros((len(unique_c1), len(unique_c2)), dtype=int)
    for i in range(n):
        row = np.where(unique_c1 == c1[i])[0][0]
        col = np.where(unique_c2 == c2[i])[0][0]
        C[row, col] += 1

    # Calculate indices
    nis = np.sum(np.sum(C, axis=1)**2)
    njs = np.sum(np.sum(C, axis=0)**2)
    t1 = comb(n, 2, exact=True)
    t2 = np.sum(C**2)
    t3 = 0.5 * (nis + njs)
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    A = t1 + t2 - t3
    D = -t2 + t3

    # Calculate indices
    if t1 == nc:
        AR = 0
    else:
        AR = (A - nc) / (t1 - nc)
    RI = A / t1
    MI = D / t1
    HI = (A - D) / t1

    return AR, RI, MI, HI
