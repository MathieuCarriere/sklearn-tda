"""
@author: Mathieu Carriere
All rights reserved
"""

import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from .hera_wasserstein import *
    from .hera_bottleneck import *
    from .kernels import *
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Cython not found--WassersteinDistance and SlicedWassersteinDistance not available")

#############################################
# Metrics ###################################
#############################################

def compute_wass_matrix(diags1, diags2, p = 1, delta = 0.001):

    num_diag1 = len(diags1)

    if np.array_equal(np.concatenate(diags1,0), np.concatenate(diags2,0)) == True:
        matrix = np.zeros((num_diag1, num_diag1))

        if USE_CYTHON:
            if np.isinf(p):
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(i+1, num_diag1):
                        matrix[i,j] = bottleneck(diags1[i], diags1[j], delta)
                        matrix[j,i] = matrix[i,j]
            else:
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(i+1, num_diag1):
                        matrix[i,j] = wasserstein(diags1[i], diags1[j], p, delta)
                        matrix[j,i] = matrix[i,j]
        else:
            print("Cython required---returning null matrix")

    else:
        num_diag2 = len(diags2)
        matrix = np.zeros((num_diag1, num_diag2))

        if USE_CYTHON:
            if np.isinf(p):
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(num_diag2):
                        matrix[i,j] = bottleneck(diags1[i], diags2[j], delta)
            else:
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(num_diag2):
                        matrix[i,j] = wasserstein(diags1[i], diags2[j], p, delta)
        else:
            print("Cython required---returning null matrix")

    return matrix

class WassersteinDistance(BaseEstimator, TransformerMixin):

    def __init__(self, wasserstein = 1, delta = 0.001):
        self.wasserstein = wasserstein
        self.delta = delta

    def fit(self, X, y = None):
        self.diagrams_ = X
        return self

    def transform(self, X):
        return compute_wass_matrix(X, self.diagrams_, self.wasserstein, self.delta)

class SlicedWassersteinDistance(BaseEstimator, TransformerMixin):

    def __init__(self, num_directions = 10):
        self.num_directions = num_directions

    def fit(self, X, y = None):
        self.diagrams_ = X
        return self

    def transform(self, X):
        if USE_CYTHON:
            Xfit = np.array(sliced_wasserstein_matrix(X, self.diagrams_, self.num_directions))
        else:
            Xfit = np.zeros((len(X), len(self.diagrams_)))
            print("Cython required---returning null matrix")
        return Xfit
