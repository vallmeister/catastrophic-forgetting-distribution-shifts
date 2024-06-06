from collections import defaultdict

import numpy as np
from sklearn import metrics


def mmd_linear(X, Z):
    XX = np.dot(X, X.T)
    ZZ = np.dot(Z, Z.T)
    XZ = np.dot(X, Z.T)
    return XX.mean() + ZZ.mean() - 2 * XZ.mean()


def mmd_rbf(X, Z, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma).mean()
    ZZ = metrics.pairwise.rbf_kernel(Z, Z, gamma).mean()
    XZ = metrics.pairwise.rbf_kernel(X, Z, gamma).mean()
    return XX + ZZ - 2 * XZ


def mmd_max_rbf(X, Z, d=128):
    GAMMA = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
             25.0, 50.0, 75.0, 100.0, d, 1.5 * d, 2 * d]
    max_mmd = 0
    for g in GAMMA:
        mmd = mmd_rbf(X, Z, g)
        max_mmd = max(max_mmd, mmd)
    return max_mmd


def total_variation_distance(p, q):
    p = defaultdict(float, p)
    q = defaultdict(float, q)
    keys = p.keys() | q.keys()
    return 0.5 * round(sum(abs(p[k] - q[k]) for k in keys), 6)


def get_average_accuracy(matrix):
    t = matrix.size(0)
    return 1 / t * sum(matrix[-1][i].item() for i in range(t))


def get_forgetting_value(matrix, i, j):
    return max(matrix[k][i].item() for k in range(j)) - matrix[j][i].item()


def get_average_forgetting_measure(matrix):
    t = matrix.size(0)
    if t > 1:
        return 1 / (t - 1) * sum(get_forgetting_value(matrix, i, t - 1) for i in range(t))
    else:
        return 0
