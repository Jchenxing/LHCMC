
from update_keth_b import *
from update_z import *
import os
os.environ["OMP_NUM_THREADS"] = "1" # Also tested with "2"
os.environ["MKL_NUM_THREADS"] = "1"


import numpy as np
from sklearn.metrics import pairwise_distances

def knn_self_representation(X, k, mode='similarity'):
    """
    X: d x N data matrix (columns are samples), will be cast to float32
    k: number of neighbors
    mode: 'similarity' (1 - distance) or 'distance' (inverse distance)
    Returns: G.T (N x N) self-representation matrix, o (N,), neighbor_idx (N x k)
    """


    N = X.shape[1]

    # 计算 pairwise 距离并强制为 float32
    dist = pairwise_distances(X.T, X.T).astype(np.float32)  # N x N

    # 计算行和（虽然未使用）
    row_sums = dist.sum(axis=1, keepdims=True).astype(np.float32)

    if mode == 'similarity':
        sim = 1.0 / (dist + np.float32(1e-8))
        np.fill_diagonal(sim, 0.0)
    elif mode == 'distance':
        sim = dist.copy()
        np.fill_diagonal(sim, np.float32(np.inf))
    else:
        raise ValueError("mode should be 'similarity' or 'distance'")

    if mode == 'distance':
        neighbor_idx = np.argsort(sim, axis=1)[:, :k]
    else:
        neighbor_idx = np.argsort(-sim, axis=1)[:, :k]

    # 最近邻距离（每行最小的非对角距离）
    max_dist = np.max(dist).astype(np.float32)
    o = np.min(dist + np.eye(N, dtype=np.float32) * max_dist, axis=1)

    # 获取每个样本的 k 近邻对应的相似度值
    row_idx = np.arange(N)[:, None]
    values = sim[row_idx, neighbor_idx]  # N x k

    # 构造 G 矩阵并归一化
    G = np.zeros_like(sim, dtype=np.float32)
    norm_values = values / (np.sum(values, axis=1, keepdims=True) + np.float32(1e-8))
    G[row_idx, neighbor_idx] = norm_values.astype(np.float32)

    return G.T, o.astype(np.float32), neighbor_idx


def main(i, alpha,beta, X, o, G, neighbor_idx):
    x = X[:,i]
    X_neighbors = X[:, neighbor_idx[i, :]]

    k = X_neighbors.shape[1]
    N = X.shape[1]
    C = (1 / k) * (X_neighbors - x[:, np.newaxis]) @ (X_neighbors - x[:, np.newaxis]).T
    """
    初始化
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    tol = 1e-8  # 阈值，避免数值精度误差
    idx = np.where(eigenvalues > tol)[0][0]  # 找到第一个非零特征值的下标
    w = eigenvectors[:, idx]
    w = w / np.linalg.norm(w)  # Normalize
    nx = X_neighbors - x[:, np.newaxis]
    o = o[i]
    b = - np.sign(np.mean((w.reshape(1, -1) @ nx))) * (o / 3)
    keth = np.maximum(0, -b * (w.reshape(1, -1) @ nx + b))
    keth = np.squeeze(keth) #k维
    keth_ = np.zeros(N,dtype=np.float32)
    keth_[neighbor_idx[i, :]] = keth #N维

    z_local = G[neighbor_idx[i, :], i]
    z = np.zeros(N,dtype=np.float32)
    z[neighbor_idx[i, :]] = z_local
    conv = True
    iter = 0
    z_old = z
    max_iter = 15
    obj_history = []

    while conv:
        # update z


        z = proximal_gradient_z_i(X,i,z_old, neighbor_idx, keth,alpha, beta)

        z_new = z


        z_local = z[neighbor_idx[i, :]]


        b, keth = optimize_b(z_local, w, x, X_neighbors, o, alpha)
        keth_full = np.zeros_like(z, dtype=np.float32)
        keth_full[neighbor_idx[i, :]] = keth
        obj_val = np.linalg.norm(x - X @ z, 2) ** 2 + (beta + alpha * keth_full) @ np.abs(z)

        obj_history.append(obj_val)
        iter += 1
        if iter >1:
            if iter >= max_iter: #or np.linalg.norm(obj_history[iter-1] - obj_history[iter-2]) <= 1e-4:#np.linalg.norm(z_new - z_old) <= 1e-3:
               conv = False
        z_old = z_new




    return i,z_new,w,b