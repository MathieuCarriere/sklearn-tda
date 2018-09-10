"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

#############################################
# Preprocessing #############################
#############################################

class BirthPersistenceTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return np.tensordot(X, np.array([[1.0, -1.0],[0.0, 1.0]]), 1)


class DiagramPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, use = False, scaler = StandardScaler()):
        self.scaler = scaler
        self.use    = use

    def fit(self, X, y = None):
        if self.use == True:
            if len(X) == 1:
                P = X[0]
            else:
                P = np.concatenate(X,0)
            self.scaler.fit(P)
        return self

    def transform(self, X):
        if self.use == True:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] > 0:
                    diag = self.scaler.transform(diag)
                Xfit.append(diag)
        else:
            Xfit = X
        return Xfit

class ProminentPoints(BaseEstimator, TransformerMixin):

    def __init__(self, use = False, num_pts = 10, threshold = -1, point_type = "upper"):
        self.num_pts    = num_pts
        self.threshold  = threshold
        self.use        = use
        self.point_type = point_type

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.use == True:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] > 0:
                    pers       = np.matmul(diag, [-1.0, 1.0])
                    idx_thresh = pers >= self.threshold
                    thresh_diag, thresh_pers  = diag[idx_thresh.flatten()], pers[idx_thresh.flatten()]
                    sort_index  = np.flip(np.argsort(thresh_pers, axis = None),0)
                    if self.point_type == "upper":
                        new_diag = thresh_diag[sort_index[:min(self.num_pts, thresh_diag.shape[0])],:]
                    if self.point_type == "lower":
                        new_diag = np.concatenate( [ thresh_diag[sort_index[min(self.num_pts, thresh_diag.shape[0]):],:], diag[~idx_thresh.flatten()] ], axis = 0)
                    Xfit.append(new_diag)
                else:
                    Xfit.append(diag)
        else:
            Xfit = X
        return Xfit

class DiagramSelector(BaseEstimator, TransformerMixin):

    def __init__(self, limit = np.inf, point_type = "finite"):
        self.limit, self.point_type = limit, point_type

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        if self.point_type == "finite":
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] != 0:
                    idx_fin = diag[:,1] != self.limit
                    Xfit.append(diag[idx_fin,:])
                else:
                    Xfit.append(diag)
        if self.point_type == "essential":
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] != 0:
                    idx_ess = diag[:,1] == self.limit
                    Xfit.append(np.reshape(diag[:,0][idx_ess],[-1,1]))
                else:
                    Xfit.append(diag[:,:1])
        return Xfit
