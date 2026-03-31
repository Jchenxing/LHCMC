import os
from scipy.io import loadmat
from sklearn.decomposition import PCA
from main import *
from myNMIACCV2 import *

import time
import os



def threshold_columns_by_ratio(Z, rho=np.float32(0.7)):
    Z_new = np.zeros_like(Z,dtype=np.float32)
    N = Z.shape[1]
    k = int(np.round(rho * N))

    for i in range(N):
        col = np.abs(Z[:, i])
        idx = np.argsort(col)[::-1]  # indices of sorted values (descending)
        top_idx = idx[:k]  # indices of top-k largest values
        Z_new[top_idx, i] = Z[top_idx, i]  # retain original values (with sign)

    return Z_new

import pandas as pd



def structure_refine(Z0, knn_graph, mu=np.float32(1.0)):
    """
    Z0: 初始 Z (n x n)，逐样本稀疏优化所得
    knn_graph: (n x n) 图邻接矩阵（可由Z0构建）
    mu: 图正则化强度
    """
    # 构造对称图拉普拉斯矩阵
    W = (knn_graph + knn_graph.T) / 2
    D = np.diag(W.sum(axis=1))
    L = D - W

    # 优化问题的闭式解： (I + mu * L)^-1 @ Z0
    A = np.eye(Z0.shape[0],dtype=np.float32) + mu * L
    Z = np.linalg.solve(A, Z0)

    # 对角线强制为 0（防止自表达）
    np.fill_diagonal(Z, 0)

    return Z


def build_knn_graph(Z0, k):
    from sklearn.neighbors import NearestNeighbors
    N = Z0.shape[0]
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(Z0)
    idx = knn.kneighbors(Z0, return_distance=False)
    graph = np.zeros((N, N),dtype=np.float32)
    for i in range(N):
        for j in idx[i, 1:]:  # exclude self
            graph[i, j] = 1
    return graph



def normalize(X):
    """标准化数据"""
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    return X

def save_results(file_path, result):
    """保存结果"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, result)


def process_dataset(dataset_path, dataset_name, numclass, k, alpha_values, beta_values):
    """处理单个数据集"""
    print(f"Processing {dataset_name}...")

    # 加载数据集
    data = loadmat(dataset_path)
    X = data['fea']#fea
    y = data['gnd']#gnd

    # PCA降维
    if X.shape[1]>60:
        pca = PCA(n_components=60)
        X_pca = pca.fit_transform(X)
        X = X_pca.T
    else:
        X = X.T
    X = normalize(X)
    X = X.astype(np.float32)
    N = X.shape[1]
    best_acc = 0
    best_idx = np.zeros(N)

    for alpha in alpha_values:
        for beta in beta_values:
            print(f"Alpha: {alpha}, Beta: {beta}")
            start_time = time.perf_counter()

            # K近邻自表示
            G, o, neighbor_idx = knn_self_representation(X, k, mode='similarity')
            G = G.T

            # 并行处理
            results = Parallel(n_jobs=-1)(
                delayed(main)(i, alpha, beta, X, o, G, neighbor_idx) for i in range(N)
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            # 拆分结果
            Z_final = np.column_stack([res[1] for res in results])  # 系数矩阵

            graph = build_knn_graph(Z_final, k)
            Z_final = structure_refine(Z_final, graph, mu=np.float32(1.0))

            # 计算并保存结果
            Z = (np.abs(Z_final) + np.abs(Z_final.T)) / 2
            # Z = (Z_final+Z_final.T)/2
            Z = Z / (np.max(Z, axis=0, keepdims=True) + 1e-8)
            Z = threshold_columns_by_ratio(Z, rho=np.float32(0.7))
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.sum(Z, axis=1)) + 1e-8))
            L_sym = np.eye(Z.shape[0], dtype=np.float32) - D_inv_sqrt @ Z @ D_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(L_sym)
            H = eigvecs[:, :numclass]
            res, idx = myNMIACCV2(H, y, numclass)
            acc, nmi, pur, ari = res
            print(res)

            df = pd.DataFrame([{
                'seq_name': dataset_name,
                'alpha': alpha,
                'beta': beta,
                'acc': acc,
                'nmi': nmi,
                'pur': pur,
                'ari': ari,
                'time_sec': elapsed,
                'k': k
            }])
            df.to_csv(f'./result20260211/result_{dataset_name}.csv', mode='a',
                      header=not os.path.exists(f'./result20260202/result_{dataset_name}.csv'), index=False)
            if 'best_acc' not in locals() or acc > best_acc:
                best_acc = acc
                best_result = {
                    'seq_name': dataset_name,
                    'alpha': alpha,
                    'beta': beta,
                    'acc': acc,
                    'nmi': nmi,
                    'pur': pur,
                    'ari': ari
                }
                best_idx = idx




datasets = [
    ('D:/USPS.mat', 'USPS', 10, 20),
]

# 参数设置
alpha_values = [2**i for i in range(0, 8)]  # 2^1, 2^2, 2^3
beta_values = [2**i for i in range(0, 9)]   # 2^1, 2^2, 2^3

# 批处理多个数据集
all_best_results = []
for dataset_path, dataset_name, numclass in datasets:
    process_dataset(dataset_path, dataset_name, numclass,k, alpha_values, beta_values)
    #all_best_results.append(best_result)

