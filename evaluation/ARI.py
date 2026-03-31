import numpy as np
from scipy.special import comb

def adjrandindex(u, v):
    '''

INPUTS
 u = the labeling as predicted by a clustering algorithm
 v = the true labeling

OUTPUTS
 adjrand = the adjusted Rand index
    '''
    n = len(u)
    ku = np.max(u)
    kv = np.max(v)
    ku = int(ku)
    kv = int(kv)
    m = np.zeros((ku, kv))
    for i in range(n):
        m[int(u[i] - 1), int(v[i] - 1)] += 1  # 因为 Python 的索引是从 0 开始的，所以需要减去 1
    mu = np.sum(m, axis=1)
    mv = np.sum(m, axis=0)
    a = 0
    for i in range(ku):
        for j in range(kv):
            if m[i, j] > 1:
                a += comb(m[i, j], 2, exact=True)
    b1 = 0
    b2 = 0
    for i in range(ku):
        if mu[i] > 1:
            b1 += comb(mu[i], 2, exact=True)
    for i in range(kv):
        if mv[i] > 1:
            b2 += comb(mv[i], 2, exact=True)
    c = comb(n, 2, exact=True)
    adjrand = (a - b1 * b2 / c) / (0.5 * (b1 + b2) - b1 * b2 / c)
    return adjrand
