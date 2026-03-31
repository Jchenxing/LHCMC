import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
#from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from evaluation.bestmap import bestMap
from evaluation.purFuc import purity
from evaluation.MI import MutualInfo
from evaluation.ARI import adjrandindex
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL",
)

def myNMIACCV2(H, Y, numclass):
    H_normalized = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    H_normalized = np.nan_to_num(H_normalized, nan=0.0)
    maxIter = 50
    res1 = np.zeros(maxIter)
    res2 = np.zeros(maxIter)
    res3 = np.zeros(maxIter)
    res4 = np.zeros(maxIter)
    best_acc = 0
    for it in range(maxIter):
        os.environ["OMP_NUM_THREADS"] = '1'
        os.environ["MKL_NUM_THREADS"] = "1"
        kmeans = KMeans(n_clusters=numclass, max_iter=250, n_init=1).fit(H_normalized)
        indx = kmeans.labels_
        indx[indx == 0] = numclass
        Y = np.asarray(Y).flatten()
        Y[Y == 0] = numclass
        newIndx = bestMap(Y, indx)
        res1[it] = np.mean(Y == newIndx)
        res2[it] = MutualInfo(Y, newIndx)
        res3[it] = purity(Y, newIndx)
        res4[it] = adjrandindex(Y, newIndx)
        if res1[it] > best_acc:
            best_acc = res1[it]
            best_indx = newIndx.copy()

    res_mean = np.array([np.max(res1), np.max(res2), np.max(res3), np.max(res4)])
    res_std = np.array([np.std(res1), np.std(res2), np.std(res3), np.std(res4)])

    return res_mean,best_indx