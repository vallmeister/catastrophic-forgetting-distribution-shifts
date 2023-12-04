import numpy as np
from numpy.linalg import norm
from sklearn import metrics


def mmd_rbf(X, Z, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    ZZ = metrics.pairwise.rbf_kernel(Z, Z, gamma)
    XZ = metrics.pairwise.rbf_kernel(X, Z, gamma)
    return XX.mean() + ZZ.mean() - 2 * XZ.mean()
