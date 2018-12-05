"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances

#############################################
# Kernel methods ############################
#############################################

class SlicedWassersteinDistance(BaseEstimator, TransformerMixin):

    def __init__(self, num_directions=10):
        self.num_directions_ = num_directions
        thetas = np.linspace(-np.pi/2, np.pi/2, num=self.num_directions_+1)[np.newaxis,:-1]
        self.lines_ = np.concatenate([np.cos(thetas), np.sin(thetas)], axis=0)

    def fit(self, X, y=None):
        self.diagrams_ = X
        self.approx_ = [np.matmul(X[i], self.lines_) for i in range(len(X))]
        diag_proj = (1./2) * np.ones((2,2))
        self.approx_diag_ = [np.matmul(np.matmul(X[i], diag_proj), self.lines_) for i in range(len(X))]
        return self

    def transform(self, X):
        Xfit = np.zeros((len(X), len(self.approx_)))
        if self.diagrams_ == X:
            for i in range(len(self.approx_)):
                for j in range(i+1, len(self.approx_)):
                    A = np.sort(np.concatenate([self.approx_[i], self.approx_diag_[j]], axis=0), axis=0)
                    B = np.sort(np.concatenate([self.approx_[j], self.approx_diag_[i]], axis=0), axis=0)
                    L1 = np.sum(np.abs(A-B), axis=0)
                    Xfit[i,j] = np.mean(L1)
                    Xfit[j,i] = Xfit[i,j]
        else:
            diag_proj = (1./2) * np.ones((2,2))
            approx = [np.matmul(X[i], self.lines_) for i in range(len(X))]
            approx_diag = [np.matmul(np.matmul(X[i], diag_proj), self.lines_) for i in range(len(X))]
            for i in range(len(approx)):
                for j in range(len(self.approx_)):
                    A = np.sort(np.concatenate([approx[i], self.approx_diag_[j]], axis=0), axis=0)
                    B = np.sort(np.concatenate([self.approx_[j], approx_diag[i]], axis=0), axis=0)
                    L1 = np.sum(np.abs(A-B), axis=0)
                    Xfit[i,j] = np.mean(L1)

        return Xfit

class SlicedWassersteinKernel(BaseEstimator, TransformerMixin):

    def __init__(self, num_directions=10, bandwidth=1.0):
        self.bandwidth_ = bandwidth
        self.sw_ = SlicedWassersteinDistance(num_directions=num_directions)

    def fit(self, X, y=None):
        self.sw_.fit(X, y)
        return self

    def transform(self, X):
        return np.exp(-self.sw_.transform(X)/self.bandwidth_)

class PersistenceWeightedGaussianKernel(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth=1.0, weight=lambda x: 1, use_pss=False):
        self.bandwidth_ = bandwidth
        self.weight_    = weight
        self.use_pss_   = use_pss

    def fit(self, X, y=None):
        self.diagrams_ = list(X)
        self.ws_ = []
        if self.use_pss_:
            for i in range(len(self.diagrams_)):
                op_D = np.tensordot(self.diagrams_[i], np.array([[0.0,1.0], [1.0,0.0]]), 1)
                self.diagrams_[i] = np.concatenate([self.diagrams_[i], op_D], 0)
        self.ws_ = [ np.array([self.weight_(self.diagrams_[i][j,:]) for j in range(self.diagrams_[i].shape[0])]) for i in range(len(self.diagrams_)) ]
        return self

    def transform(self, X):
        Xp = list(X)
        if self.use_pss_:
            for i in range(len(Xp)):
                op_X = np.tensordot(Xp[i], np.array([[0.0,1.0], [1.0,0.0]]), 1)
                Xp[i] = np.concatenate([Xp[i], op_X], 0)
        Xfit = np.zeros((len(Xp), len(self.diagrams_)))
        if self.diagrams_ == Xp:
            for i in range(len(self.diagrams_)):
                for j in range(i+1, len(self.diagrams_)):
                    W = np.matmul(self.ws_[i][:,np.newaxis], self.ws_[j][np.newaxis,:])
                    E = (1./(np.sqrt(2*np.pi)*self.bandwidth_)) * np.exp(-np.square(pairwise_distances(self.diagrams_[i], self.diagrams_[j]))/(2*np.square(self.bandwidth_)))
                    Xfit[i,j] = np.sum(np.multiply(W, E))
                    Xfit[j,i] = X[i,j]
        else:
            ws = [ np.array([self.weight_(Xp[i][j,:]) for j in range(Xp[i].shape[0])]) for i in range(len(Xp)) ]
            for i in range(len(Xp)):
                for j in range(len(self.diagrams_)):
                    W = np.matmul(ws[i][:,np.newaxis], self.ws_[j][np.newaxis,:])
                    E = (1./(np.sqrt(2*np.pi)*self.bandwidth_)) * np.exp(-np.square(pairwise_distances(Xp[i], self.diagrams_[j]))/(2*np.square(self.bandwidth_)))
                    Xfit[i,j] = np.sum(np.multiply(W, E))
        
        return Xfit

class PersistenceScaleSpaceKernel(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth=1.0):
        self.pwg_ = PersistenceWeightedGaussianKernel(bandwidth=bandwidth, weight=lambda x: 1 if x[1] >= x[0] else -1, use_pss=True)

    def fit(self, X, y=None):
        self.pwg_.fit(X,y)
        return self

    def transform(self, X):
        return self.pwg_.transform(X)
