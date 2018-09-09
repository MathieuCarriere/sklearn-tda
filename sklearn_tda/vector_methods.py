"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors     import DistanceMetric

from .preprocessing import DiagramPreprocessor

try:
    from .vectors import *
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

#############################################
# Finite Vectorization methods ##############
#############################################

class PersistenceImage(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0, weight = lambda x: 1,
                       resolution = [20,20], im_range = [np.nan, np.nan, np.nan, np.nan]):
        self.bandwidth, self.weight = bandwidth, weight
        self.resolution, self.im_range = resolution, im_range

    def fit(self, X, y = None):
        if np.isnan(self.im_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.im_range = [mx, Mx, my, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                image = np.array(persistence_image(diagram, self.im_range[0], self.im_range[1], self.resolution[0], self.im_range[2], self.im_range[3], self.resolution[1], self.bandwidth, self.weight))

            else:

                w = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    w[j] = self.weight(diagram[j,:])

                x_values, y_values = np.linspace(self.im_range[0], self.im_range[1], self.resolution[0]), np.linspace(self.im_range[2], self.im_range[3], self.resolution[1])
                Xs, Ys = np.tile((diagram[:,0][:,np.newaxis,np.newaxis]-x_values[np.newaxis,np.newaxis,:]),[1,self.resolution[1],1]), np.tile(diagram[:,1][:,np.newaxis,np.newaxis]-y_values[np.newaxis,:,np.newaxis],[1,1,self.resolution[0]])
                image = np.tensordot(w, np.exp((-np.square(Xs)-np.square(Ys))/(2*self.bandwidth*self.bandwidth))/(self.bandwidth*np.sqrt(2*np.pi)), 1)

            Xfit.append(image.flatten()[np.newaxis,:])

        return np.concatenate(Xfit,0)

class Landscape(BaseEstimator, TransformerMixin):

    def __init__(self, num_landscapes = 5, resolution = 100, ls_range = [np.nan, np.nan]):
        self.num_landscapes, self.resolution, self.ls_range = num_landscapes, resolution, ls_range

    def fit(self, X, y = None):
        if np.isnan(self.ls_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.ls_range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.ls_range[0], self.ls_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(landscape(diagram, self.num_landscapes, self.ls_range[0], self.ls_range[1], self.resolution)).flatten()[np.newaxis,:])

            else:

                ls = np.zeros([self.num_landscapes, self.resolution])

                events = []
                for j in range(self.resolution):
                    events.append([])

                for j in range(num_pts_in_diag):
                    [px,py] = diagram[j,:]
                    min_idx = np.minimum(np.maximum(np.ceil((px          - self.ls_range[0]) / step_x).astype(int), 0), self.resolution)
                    mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.ls_range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py          - self.ls_range[0]) / step_x).astype(int), 0), self.resolution)

                    if min_idx < self.resolution and max_idx > 0:

                        landscape_value = self.ls_range[0] + min_idx * step_x - px
                        for k in range(min_idx, mid_idx):
                            events[k].append(landscape_value)
                            landscape_value += step_x

                        landscape_value = py - self.ls_range[0] - mid_idx * step_x
                        for k in range(mid_idx, max_idx):
                            events[k].append(landscape_value)
                            landscape_value -= step_x

                for j in range(self.resolution):
                    events[j].sort(reverse = True)
                    for k in range( min(self.num_landscapes, len(events[j])) ):
                        ls[k,j] = events[j][k]

                Xfit.append(np.sqrt(2)*np.reshape(ls,[1,-1]))

        return np.concatenate(Xfit,0)

class Silhouette(BaseEstimator, TransformerMixin):

    def __init__(self, weight = lambda x: 1, resolution = 100, sh_range = [np.nan, np.nan]):
        self.weight, self.resolution, self.sh_range = weight, resolution, sh_range

    def fit(self, X, y = None):
        if np.isnan(self.sh_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.sh_range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sh_range[0], self.sh_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(silhouette(diagram, self.sh_range[0], self.sh_range[1], self.resolution, self.weight))[np.newaxis,:])

            else:

                sh, weights = np.zeros(self.resolution), np.zeros(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    weights[j] = self.weight(diagram[j,:])
                total_weight = np.sum(weights)

                for j in range(num_pts_in_diag):

                    [px,py] = diagram[j,:]
                    weight  = weights[j] / total_weight
                    min_idx = np.minimum(np.maximum(np.ceil((px          - self.sh_range[0]) / step_x).astype(int), 0), self.resolution)
                    mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.sh_range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py          - self.sh_range[0]) / step_x).astype(int), 0), self.resolution)

                    if min_idx < self.resolution and max_idx > 0:

                        silhouette_value = self.sh_range[0] + min_idx * step_x - px
                        for k in range(min_idx, mid_idx):
                            sh[k] += weight * silhouette_value
                            silhouette_value += step_x

                        silhouette_value = py - self.sh_range[0] - mid_idx * step_x
                        for k in range(mid_idx, max_idx):
                            sh[k] += weight * silhouette_value
                            silhouette_value -= step_x

                Xfit.append(np.reshape(np.sqrt(2) * sh, [1,-1]))

        return np.concatenate(Xfit, 0)

class BettiCurve(BaseEstimator, TransformerMixin):

    def __init__(self, resolution = 100, bc_range = [np.nan, np.nan]):
        self.resolution, self.bc_range = resolution, bc_range

    def fit(self, X, y = None):
        if np.isnan(self.bc_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.bc_range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.bc_range[0], self.bc_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(betti_curve(diagram, self.bc_range[0], self.bc_range[1], self.resolution))[np.newaxis,:])

            else:

                bc =  np.zeros(self.resolution)
                for j in range(num_pts_in_diag):
                    [px,py] = diagram[j,:]
                    min_idx = np.minimum(np.maximum(np.ceil((px - self.bc_range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py - self.bc_range[0]) / step_x).astype(int), 0), self.resolution)
                    for k in range(min_idx, max_idx):
                        bc[k] += 1

                Xfit.append(np.reshape(bc,[1,-1]))

        return np.concatenate(Xfit, 0)

class TopologicalVector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold = 10):
        self.threshold = threshold

    def fit(self, X, y = None):
        return self

    def transform(self, X):

        num_diag = len(X)
        Xfit = np.zeros([num_diag, self.threshold])

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]
            pers = 0.5 * np.matmul(diagram, np.array([[-1.0],[1.0]]))
            min_pers = np.minimum(pers,np.transpose(pers))
            distances = DistanceMetric.get_metric("chebyshev").pairwise(diagram)
            vect = np.flip(np.sort(np.triu(np.minimum(distances, min_pers)), axis = None), 0)
            dim = np.minimum(len(vect), self.threshold)
            Xfit[i, :dim] = vect[:dim]

        return Xfit

