import numpy as np
from sklearn import metrics


def mmd_linear(X, Z):
    XX = np.dot(X, X.T)
    ZZ = np.dot(Z, Z.T)
    XZ = np.dot(X, Z.T)
    return XX.mean() + ZZ.mean() - 2 * XZ.mean()


def mmd_rbf(X, Z, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    ZZ = metrics.pairwise.rbf_kernel(Z, Z, gamma)
    XZ = metrics.pairwise.rbf_kernel(X, Z, gamma)
    return XX.mean() + ZZ.mean() - 2 * XZ.mean()


def mmd_max_rbf(X, Z, d=128):
    Gamma = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
             25.0, 50.0, 75.0, 100.0, d, 1.5 * d, 2 * d]
    max_mmd = 0
    for g in Gamma:
        mmd = mmd_rbf(X, Z, g)
        max_mmd = max(max_mmd, mmd)
    return max_mmd


def total_variation_distance(P, Q):
    c = len(P)
    return 0.5 * sum(abs(P[k] - Q[k]) for k in range(c))
