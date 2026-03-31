import numpy as np

def MutualInfo(L1, L2):
    '''
    这个函数计算了两个输入向量的互信息，这里的 L1 和 L2 是一维数组，表示对应的标签。
    '''
    L1 = np.ravel(L1)
    L2 = np.ravel(L2)
    if L1.size != L2.size:
        raise ValueError('size(L1) must == size(L2)')

    Label = np.unique(L1)
    nClass = len(Label)

    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    if nClass2 < nClass:
        # smooth
        L1 = np.concatenate((L1, Label))
        L2 = np.concatenate((L2, Label))
    elif nClass2 > nClass:
        # smooth
        L1 = np.concatenate((L1, Label2))
        L2 = np.concatenate((L2, Label2))

    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == Label[i]) & (L2 == Label[j]))
    sumG = np.sum(G)

    P1 = np.sum(G, axis=1)
    P1 = P1 / sumG
    P2 = np.sum(G, axis=0)
    P2 = P2 / sumG

    if np.sum(P1 == 0) > 0 or np.sum(P2 == 0) > 0:
        # smooth
        raise ValueError('Smooth fail!')
    else:
        H1 = np.sum(-P1 * np.log2(P1))
        H2 = np.sum(-P2 * np.log2(P2))
        P12 = G / sumG
        PPP = P12 / np.tile(P2, (nClass, 1)) / np.tile(P1[:, np.newaxis], (1, nClass))
        PPP[np.abs(PPP) < 1e-12] = 1
        MI = np.sum(P12 * np.log2(PPP))
        MIhat = MI / max(H1, H2)
        MIhat = np.real(MIhat)
    return MIhat
