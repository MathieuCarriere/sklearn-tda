"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors     import DistanceMetric

from .preprocessing import DiagramPreprocessor

#############################################
# Finite Vectorization methods ##############
#############################################

class PersistenceImage(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth=1.0, weight=lambda x: 1, resolution=[20,20], im_range=[np.nan, np.nan, np.nan, np.nan]):
        self.bandwidth_, self.weight_ = bandwidth, weight
        self.resolution_, self.im_range_ = resolution, im_range

    def fit(self, X, y=None):
        if np.isnan(self.im_range_[0]):
            pre = DiagramPreprocessor(use=True, scalers=[([0,1], MinMaxScaler())]).fit(X,y)
            [mx,my],[Mx,My] = pre.scalers[0][1].data_min_, pre.scalers[0][1].data_max_
            self.im_range_ = [mx, Mx, my, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            w = np.ones(num_pts_in_diag)
            for j in range(num_pts_in_diag):
                w[j] = self.weight_(diagram[j,:])

            x_values, y_values = np.linspace(self.im_range_[0], self.im_range_[1], self.resolution_[0]), np.linspace(self.im_range_[2], self.im_range_[3], self.resolution_[1])
            Xs, Ys = np.tile((diagram[:,0][:,np.newaxis,np.newaxis]-x_values[np.newaxis,np.newaxis,:]),[1,self.resolution_[1],1]), np.tile(diagram[:,1][:,np.newaxis,np.newaxis]-y_values[np.newaxis,:,np.newaxis],[1,1,self.resolution_[0]])
            image = np.tensordot(w, np.exp((-np.square(Xs)-np.square(Ys))/(2*np.square(self.bandwidth_)))/(self.bandwidth_*np.sqrt(2*np.pi)), 1)

            Xfit.append(image.flatten()[np.newaxis,:])

        return np.concatenate(Xfit,0)

class Landscape(BaseEstimator, TransformerMixin):

    def __init__(self, num_landscapes=5, resolution=100, ls_range=[np.nan, np.nan]):
        self.num_landscapes_, self.resolution_, self.ls_range_ = num_landscapes, resolution, ls_range

    def fit(self, X, y=None):
        if np.isnan(self.ls_range_[0]):
            pre = DiagramPreprocessor(use=True, scalers=[([0,1], MinMaxScaler())]).fit(X,y)
            [mx,my],[Mx,My] = pre.scalers[0][1].data_min_, pre.scalers[0][1].data_max_
            self.ls_range_ = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.ls_range_[0], self.ls_range_[1], self.resolution_)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            ls = np.zeros([self.num_landscapes_, self.resolution_])

            events = []
            for j in range(self.resolution_):
                events.append([])

            for j in range(num_pts_in_diag):
                [px,py] = diagram[j,:]
                min_idx = np.minimum(np.maximum(np.ceil((px          - self.ls_range_[0]) / step_x).astype(int), 0), self.resolution_)
                mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.ls_range_[0]) / step_x).astype(int), 0), self.resolution_)
                max_idx = np.minimum(np.maximum(np.ceil((py          - self.ls_range_[0]) / step_x).astype(int), 0), self.resolution_)

                if min_idx < self.resolution_ and max_idx > 0:

                    landscape_value = self.ls_range_[0] + min_idx * step_x - px
                    for k in range(min_idx, mid_idx):
                        events[k].append(landscape_value)
                        landscape_value += step_x

                    landscape_value = py - self.ls_range_[0] - mid_idx * step_x
                    for k in range(mid_idx, max_idx):
                        events[k].append(landscape_value)
                        landscape_value -= step_x

            for j in range(self.resolution_):
                events[j].sort(reverse=True)
                for k in range( min(self.num_landscapes_, len(events[j])) ):
                    ls[k,j] = events[j][k]

            Xfit.append(np.sqrt(2)*np.reshape(ls,[1,-1]))

        return np.concatenate(Xfit,0)

class Silhouette(BaseEstimator, TransformerMixin):

    def __init__(self, weight=lambda x: 1, resolution=100, sh_range=[np.nan, np.nan]):
        self.weight_, self.resolution_, self.sh_range_ = weight, resolution, sh_range

    def fit(self, X, y=None):
        if np.isnan(self.sh_range_[0]) == True:
            pre = DiagramPreprocessor(use=True, scalers=[([0,1], MinMaxScaler())]).fit(X,y)
            [mx,my],[Mx,My] = pre.scalers[0][1].data_min_, pre.scalers[0][1].data_max_
            self.sh_range_ = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sh_range_[0], self.sh_range_[1], self.resolution_)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            sh, weights = np.zeros(self.resolution_), np.zeros(num_pts_in_diag)
            for j in range(num_pts_in_diag):
                weights[j] = self.weight_(diagram[j,:])
            total_weight = np.sum(weights)

            for j in range(num_pts_in_diag):

                [px,py] = diagram[j,:]
                weight  = weights[j] / total_weight
                min_idx = np.minimum(np.maximum(np.ceil((px          - self.sh_range_[0]) / step_x).astype(int), 0), self.resolution_)
                mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.sh_range_[0]) / step_x).astype(int), 0), self.resolution_)
                max_idx = np.minimum(np.maximum(np.ceil((py          - self.sh_range_[0]) / step_x).astype(int), 0), self.resolution_)

                if min_idx < self.resolution_ and max_idx > 0:

                    silhouette_value = self.sh_range_[0] + min_idx * step_x - px
                    for k in range(min_idx, mid_idx):
                        sh[k] += weight * silhouette_value
                        silhouette_value += step_x

                    silhouette_value = py - self.sh_range_[0] - mid_idx * step_x
                    for k in range(mid_idx, max_idx):
                        sh[k] += weight * silhouette_value
                        silhouette_value -= step_x

            Xfit.append(np.reshape(np.sqrt(2) * sh, [1,-1]))

        return np.concatenate(Xfit, 0)

class BettiCurve(BaseEstimator, TransformerMixin):

    def __init__(self, resolution=100, bc_range=[np.nan, np.nan]):
        self.resolution_, self.bc_range_ = resolution, bc_range

    def fit(self, X, y=None):
        if np.isnan(self.bc_range_[0]):
            pre = DiagramPreprocessor(use=True, scalers=[([0,1], MinMaxScaler())]).fit(X,y)
            [mx,my],[Mx,My] = pre.scalers[0][1].data_min_, pre.scalers[0][1].data_max_
            self.bc_range_ = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.bc_range_[0], self.bc_range_[1], self.resolution_)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            bc =  np.zeros(self.resolution_)
            for j in range(num_pts_in_diag):
                [px,py] = diagram[j,:]
                min_idx = np.minimum(np.maximum(np.ceil((px - self.bc_range_[0]) / step_x).astype(int), 0), self.resolution_)
                max_idx = np.minimum(np.maximum(np.ceil((py - self.bc_range_[0]) / step_x).astype(int), 0), self.resolution_)
                for k in range(min_idx, max_idx):
                    bc[k] += 1

            Xfit.append(np.reshape(bc,[1,-1]))

        return np.concatenate(Xfit, 0)

class TopologicalVector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=10):
        self.threshold_ = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if self.threshold_ == -1:
            thresh = np.array([X[i].shape[0] for i in range(len(X))]).max()
        else:
            thresh = self.threshold_

        num_diag = len(X)
        Xfit = np.zeros([num_diag, thresh])

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]
            pers = 0.5 * np.matmul(diagram, np.array([[-1.0],[1.0]]))
            min_pers = np.minimum(pers,np.transpose(pers))
            distances = DistanceMetric.get_metric("chebyshev").pairwise(diagram)
            vect = np.flip(np.sort(np.triu(np.minimum(distances, min_pers)), axis=None), 0)
            dim = min(len(vect), thresh)
            Xfit[i, :dim] = vect[:dim]

        return Xfit

class ComplexPolynomial(BaseEstimator, TransformerMixin):

    def __init__(self, F="R", threshold=10):
        self.threshold_, self.F_ = threshold, F

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if self.threshold_ == -1:
            thresh = np.array([X[i].shape[0] for i in range(len(X))]).max()
        else:
            thresh = self.threshold_

        Xfit = np.zeros([len(X), thresh]) + 1j * np.zeros([len(X), thresh])
        for d in range(len(X)):
            D, N = X[d], X[d].shape[0]
            if self.F_ == "R":
                roots = D[:,0] + 1j * D[:,1]
            elif self.F_ == "S":
                alpha = np.linalg.norm(D, axis=1)
                alpha = np.where(alpha==0, np.ones(N), alpha)
                roots = np.multiply( np.multiply(  (D[:,0]+1j*D[:,1]), (D[:,1]-D[:,0])  ), 1./(np.sqrt(2)*alpha) )
            elif self.F_ == "T":
                alpha = np.linalg.norm(D, axis=1)
                roots = np.multiply(  (D[:,1]-D[:,0])/2, np.cos(alpha) - np.sin(alpha) + 1j * (np.cos(alpha) + np.sin(alpha))  )
            coeff = [0] * (N+1)
            coeff[N] = 1
            for i in range(1, N+1): 
                for j in range(N-i-1, N): 
                    coeff[j] += ((-1) * roots[i-1] * coeff[j+1])   
            coeff = np.array(coeff[::-1])[1:]
            Xfit[d, :min(thresh, coeff.shape[0])] = coeff[:min(thresh, coeff.shape[0])]
        return Xfit
