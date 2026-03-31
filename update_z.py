import numpy as np

def soft_threshold(x, theta):
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0)


def proximal_gradient_z_i(X, i, z_old, idx, keth, alpha, beta,
                          eta=np.float32(1e-5), tol=np.float32(1e-2), nonnegative=False):
    """
    带稀疏项和非负约束的近端梯度更新
    X: d × n
    i: 当前样本索引
    z_old: 初始系数 (n,)
    idx: 邻居索引矩阵 (n × k)
    keth: 第 i 行邻居的权重 (k,)
    alpha, beta: 超参数
    """
    x = X[:, i]
    z = z_old.copy().astype(np.float32)
    keth_ = np.zeros_like(z, dtype=np.float32)
    keth_[idx[i, :]] = keth
    max_iter = 50

    obj_old = np.linalg.norm(x - X @ z) ** 2 + (beta + alpha * keth_) @ np.abs(z)

    for t in range(max_iter):
        grad = 2 * X.T @ (X @ z - x)
        theta = eta * (beta + alpha * keth_)

        if nonnegative:
            # 带非负约束的 proximal 操作
            z_new = np.maximum(0, z - eta * grad - theta)
        else:
            # 标准 soft-thresholding
            u = z - eta * grad
            z_new = soft_threshold(u, theta)

        obj_new = np.linalg.norm(x - X @ z_new) ** 2 + (beta + alpha * keth_) @ np.abs(z_new)
        if abs(obj_new - obj_old) < tol:
            break

        obj_old = obj_new
        z = z_new

    return z
