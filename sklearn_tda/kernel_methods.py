"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from .kernels import *
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Cython not found--SlicedWassersteinKernel, PersistenceWeightedGaussianKernel and PersistenceScaleSpaceKernel not available")

#############################################
# Kernel methods ############################
#############################################

class SlicedWassersteinKernel(BaseEstimator, TransformerMixin):

    def __init__(self, num_directions = 10, bandwidth = 1.0):
        self.num_directions = num_directions
        self.bandwidth = bandwidth

    def fit(self, X, y = None):
        self.diagrams_ = list(X)
        return self

    def transform(self, X):
        if USE_CYTHON:
            Xfit = np.array(sliced_wasserstein_kernel_matrix(X, self.diagrams_, self.bandwidth, self.num_directions))
        else:
            Xfit = np.zeros((X.shape[0], self.diagrams_.shape[0]))
            print("Cython required---returning null matrix")
        return Xfit

class PersistenceWeightedGaussianKernel(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0, weight = lambda x: 1, use_pss = False):
        self.bandwidth = bandwidth
        self.weight    = weight
        self.use_pss   = use_pss

    def fit(self, X, y = None):
        self.diagrams_ = list(X)
        if self.use_pss == True:
            for i in range(len(self.diagrams_)):
                op_D = np.tensordot(self.diagrams_[i], np.array([[0.0,1.0],[1.0,0.0]]), 1)
                self.diagrams_[i] = np.concatenate([self.diagrams_[i], op_D], 0)
        return self

    def transform(self, X):
        Xp = list(X)
        if self.use_pss:
            for i in range(len(Xp)):
                op_X = np.tensordot(Xp[i], np.array([[0.0,1.0],[1.0,0.0]]), 1)
                Xp[i] = np.concatenate([Xp[i], op_X], 0)
        if USE_CYTHON:
            Xfit = np.array(persistence_weighted_gaussian_kernel_matrix(Xp, self.diagrams_, self.bandwidth, self.weight))
        else:
            Xfit = np.zeros((X.shape[0], self.diagrams_.shape[0]))
            print("Cython required---returning null matrix")
        return Xfit

class PersistenceScaleSpaceKernel(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0):
        self.PWG = PersistenceWeightedGaussianKernel(bandwidth = bandwidth, weight = lambda x: 1 if x[1] >= x[0] else -1, use_pss = True)

    def fit(self, X, y = None):
        self.PWG.fit(X,y)
        return self

    def transform(self, X):
        return self.PWG.transform(X)
