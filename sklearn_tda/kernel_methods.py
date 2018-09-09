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

#############################################
# Kernel methods ############################
#############################################

def mergeSorted(a, b):
    l = []
    while a and b:
        if a[0] < b[0]:
            l.append(a.pop(0))
        else:
            l.append(b.pop(0))
    return l + a + b

class SlicedWasserstein(BaseEstimator, TransformerMixin):

    def __init__(self, num_directions = 10, bandwidth = 1.0):
        self.num_directions = num_directions
        self.bandwidth = bandwidth

    def fit(self, X, y = None):

        if USE_CYTHON == False:

            num_diag = len(X)
            angles = np.linspace(-np.pi/2, np.pi/2, self.num_directions + 1)
            self.step_angle_ = angles[1] - angles[0]
            self.thetas_ = np.concatenate([np.cos(angles[:-1])[np.newaxis,:], np.sin(angles[:-1])[np.newaxis,:]], 0)

            self.proj_, self.proj_delta_ = [], []
            for i in range(num_diag):

                diagram = X[i]
                diag_thetas, list_proj = np.tensordot(diagram, self.thetas_, 1), []
                for j in range(self.num_directions):
                    list_proj.append( list(np.sort(diag_thetas[:,j])) )
                self.proj_.append(list_proj)

                diagonal_diagram = np.tensordot(diagram, np.array([[0.5,0.5],[0.5,0.5]]), 1)
                diag_thetas, list_proj_delta = np.tensordot(diagonal_diagram, self.thetas_, 1), []
                for j in range(self.num_directions):
                    list_proj_delta.append( list(np.sort(diag_thetas[:,j])) )
                self.proj_delta_.append(list_proj_delta)

        self.diagrams_ = list(X)

        return self

    def transform(self, X):

        if USE_CYTHON == False:

            num_diag1 = len(self.proj_)

            if np.array_equal(np.concatenate(self.diagrams_,0), np.concatenate(X,0)) == True:

                Xfit = np.zeros( [num_diag1, num_diag1] )
                for i in range(num_diag1):
                    for j in range(i, num_diag1):

                        L1, L2 = [], []
                        for k in range(self.num_directions):
                            ljk, ljkd, lik, likd = list(self.proj_[j][k]), list(self.proj_delta_[j][k]), list(self.proj_[i][k]), list(self.proj_delta_[i][k])
                            L1.append( np.array(mergeSorted(ljk, likd))[:,np.newaxis] )
                            L2.append( np.array(mergeSorted(lik, ljkd))[:,np.newaxis] )
                        L1, L2 = np.concatenate(L1,1), np.concatenate(L2,1)

                        Xfit[i,j] = np.sum(self.step_angle_*np.sum(np.abs(L1-L2),0)/np.pi)
                        Xfit[j,i] = Xfit[i,j]

                Xfit =  np.exp(-Xfit/(2*self.bandwidth*self.bandwidth))

            else:

                num_diag2 = len(X)
                proj, proj_delta = [], []
                for i in range(num_diag2):

                    diagram = X[i]
                    diag_thetas, list_proj = np.tensordot(diagram, self.thetas_, 1), []
                    for j in range(self.num_directions):
                        list_proj.append( list(np.sort(diag_thetas[:,j])) )
                    proj.append(list_proj)

                    diagonal_diagram = np.tensordot(diagram, np.array([[0.5,0.5],[0.5,0.5]]), 1)
                    diag_thetas, list_proj_delta = np.tensordot(diagonal_diagram, self.thetas_, 1), []
                    for j in range(self.num_directions):
                        list_proj_delta.append( list(np.sort(diag_thetas[:,j])) )
                    proj_delta.append(list_proj_delta)

                Xfit = np.zeros( [num_diag2, num_diag1] )
                for i in range(num_diag2):
                    for j in range(num_diag1):

                        L1, L2 = [], []
                        for k in range(self.num_directions):
                            ljk, ljkd, lik, likd = list(self.proj_[j][k]), list(self.proj_delta_[j][k]), list(proj[i][k]), list(proj_delta[i][k])
                            L1.append( np.array(mergeSorted(ljk, likd))[:,np.newaxis] )
                            L2.append( np.array(mergeSorted(lik, ljkd))[:,np.newaxis] )
                        L1, L2 = np.concatenate(L1,1), np.concatenate(L2,1)

                        Xfit[i,j] = np.sum(self.step_angle_*np.sum(np.abs(L1-L2),0)/np.pi)

                Xfit =  np.exp(-Xfit/(2*self.bandwidth*self.bandwidth))

        else:

            Xfit = np.array(sliced_wasserstein_matrix(X, self.diagrams_, self.bandwidth, self.num_directions))

        return Xfit

class PersistenceWeightedGaussian(BaseEstimator, TransformerMixin):

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

        if USE_CYTHON == False:
            self.w_ = []
            for i in range(len(self.diagrams_)):
                num_pts_in_diag = self.diagrams_[i].shape[0]
                w = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    w[j] = self.weight(self.diagrams_[i][j,:])
                self.w_.append(w)

        return self

    def transform(self, X):

        Xp = list(X)
        if self.use_pss == True:
            for i in range(len(Xp)):
                op_X = np.tensordot(Xp[i], np.array([[0.0,1.0],[1.0,0.0]]), 1)
                Xp[i] = np.concatenate([Xp[i], op_X], 0)

        if USE_CYTHON == True:

            Xfit = np.array(persistence_weighted_gaussian_matrix(Xp, self.diagrams_, self.bandwidth, self.weight))

        else:

            num_diag1 = len(self.w_)

            if np.array_equal(np.concatenate(Xp,0), np.concatenate(self.diagrams_,0)) == True:

                Xfit = np.zeros([num_diag1, num_diag1])

                for i in range(num_diag1):
                    for j in range(i,num_diag1):

                        d1x, d1y, d2x, d2y = self.diagrams_[i][:,0][:,np.newaxis], self.diagrams_[i][:,1][:,np.newaxis], self.diagrams_[j][:,0][np.newaxis,:], self.diagrams_[j][:,1][np.newaxis,:]
                        Xfit[i,j] = np.tensordot(self.w_[j], np.tensordot(self.w_[i], np.exp( -(np.square(d1x-d2x) + np.square(d1y-d2y)) / (2*self.bandwidth*self.bandwidth)) / (self.bandwidth*np.sqrt(2*np.pi)), 1), 1)
                        Xfit[j,i] = Xfit[i,j]
            else:

                num_diag2 = len(Xp)
                w = []
                for i in range(num_diag2):
                    num_pts_in_diag = Xp[i].shape[0]
                    we = np.ones(num_pts_in_diag)
                    for j in range(num_pts_in_diag):
                        we[j] = self.weight(Xp[i][j,:])
                    w.append(we)

                Xfit = np.zeros([num_diag2, num_diag1])

                for i in range(num_diag2):
                    for j in range(num_diag1):

                        d1x, d1y, d2x, d2y = Xp[i][:,0][:,np.newaxis], Xp[i][:,1][:,np.newaxis], self.diagrams_[j][:,0][np.newaxis,:], self.diagrams_[j][:,1][np.newaxis,:]
                        Xfit[i,j] = np.tensordot(self.w_[j], np.tensordot(w[i], np.exp( -(np.square(d1x-d2x) + np.square(d1y-d2y)) / (2*self.bandwidth*self.bandwidth)) / (self.bandwidth*np.sqrt(2*np.pi)), 1), 1)

        return Xfit

class PersistenceScaleSpace(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0):
        self.PWG = PersistenceWeightedGaussian(bandwidth = bandwidth, weight = lambda x: 1 if x[1] >= x[0] else -1, use_pss = True)

    def fit(self, X, y = None):
        self.PWG.fit(X,y)
        return self

    def transform(self, X):
        return self.PWG.transform(X)
