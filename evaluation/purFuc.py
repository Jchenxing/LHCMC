import numpy as np

def purity(Y, predY):
    predLidx = np.unique(predY)
    pred_classnum = len(predLidx)
    # purity
    correnum = 0
    for ci in range(pred_classnum):
        incluster = Y[predY == predLidx[ci]]
        inclunub = np.histogram(incluster, bins=np.arange(1, max(incluster)+2))[0]
        if len(inclunub)>0:
            correnum = correnum + max(inclunub)
        else:
            correnum += 0
    Purity = correnum / len(predY)
    return Purity
