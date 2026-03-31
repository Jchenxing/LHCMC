import numpy as np
#from evaluation.hungarian import hungarian
from scipy.optimize import linear_sum_assignment

'''
def bestMap(L1, L2):
    """
    bestmap: permute labels of L2 to match L1 as good as possible

    Parameters:
    L1 : numpy.ndarray
        The first label array.
    L2 : numpy.ndarray
        The second label array.

    Returns:
    newL2 : numpy.ndarray
        The permuted labels of L2 to match L1 as good as possible.
    """
    L1 = np.array(L1).reshape(-1)
    L2 = np.array(L2).reshape(-1)

    if L1.size != L2.size:
        raise ValueError('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))

    c, t = hungarian(-G)
    newL2 = np.zeros_like(L2)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]

    return newL2


def bestMap( ground_truth_labels,cluster_labels):
    """
    Align cluster labels with ground truth labels.

    Arguments:
        cluster_labels: array-like, cluster labels
        ground_truth_labels: array-like, ground truth labels

    Returns:
        aligned_cluster_labels: array-like, aligned cluster labels
    """
    num_clusters = len(np.unique(cluster_labels))
    num_ground_truth = len(np.unique(ground_truth_labels))
    aligned_cluster_labels = np.zeros_like(cluster_labels)

    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        max_overlap = -1
        best_match_label = -1
        for j in range(num_ground_truth):
            ground_truth_indices = np.where(ground_truth_labels == j)[0]
            overlap = len(np.intersect1d(cluster_indices, ground_truth_indices))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_label = j
        aligned_cluster_labels[cluster_labels == i] = best_match_label

    return aligned_cluster_labels
'''


def bestMap(L1, L2):
    """
    Permute labels of L2 to match L1 as good as possible.

    Arguments:
        L1: array-like, true labels
        L2: array-like, predicted labels

    Returns:
        newL2: array-like, permuted labels of L2
    """
    L1 = np.asarray(L1).flatten()
    L2 = np.asarray(L2).flatten()
    if L1.size != L2.size:
        raise ValueError('size(L1) must == size(L2)')

    unique_labels1, counts1 = np.unique(L1, return_counts=True)
    unique_labels2, counts2 = np.unique(L2, return_counts=True)

    n_class1 = len(unique_labels1)
    n_class2 = len(unique_labels2)
    n_class = max(n_class1, n_class2)

    G = np.zeros((n_class, n_class))
    for i in range(n_class1):
        ind_cla1 = L1 == unique_labels1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(n_class2):
            ind_cla2 = L2 == unique_labels2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)
    '''
    for i in range(n_class1):
        for j in range(n_class2):
            G[i, j] = np.sum((L1 == unique_labels1[i]) & (L2 == unique_labels2[j]))
    '''
    row_ind, col_ind = linear_sum_assignment(-G)
    newL2 = np.zeros_like(L2)
    for i in range(n_class2):
        newL2[L2 == unique_labels2[col_ind[i]]] = unique_labels1[row_ind[i]]

    return newL2