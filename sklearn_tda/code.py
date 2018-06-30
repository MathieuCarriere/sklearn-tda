# -*- coding: utf-8 -*-
"""
@author: Mathieu Carrière
All rights reserved
"""

import numpy             as np
from   scipy.signal      import convolve2d

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors     import DistanceMetric

try:
    from vectors import *
    from kernels import *
    from hera_wasserstein import *
    from hera_bottleneck import *
    USE_CYTHON = True
    print("Cython found")

except ImportError:
    USE_CYTHON = False
    print("Cython not found")

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

    def __init__(self, use = False, num_pts = 10, threshold = -1):
        self.num_pts   = num_pts
        self.threshold = threshold
        self.use       = use

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.use == True:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag         = X[i]
                pers         = np.matmul(diag, [-1.0, 1.0])
                idx_thresh   = pers >= self.threshold
                thresh_diag, thresh_pers  = diag[idx_thresh.flatten()], pers[idx_thresh.flatten()]
                sort_index   = np.flip(np.argsort(thresh_pers, axis = None),0)
                sorted_diag  = thresh_diag[sort_index[:min(self.num_pts, diag.shape[0])],:]
                Xfit.append(sorted_diag)
        else:
            Xfit = X
        return Xfit

class FiniteSelector(BaseEstimator, TransformerMixin):

    def __init__(self, limit = np.inf):
        self.limit = limit

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        for i in range(num_diag):
            diag = X[i]
            idx_fin = diag[:,1] != self.limit
            Xfit.append(diag[idx_fin,:])
        return Xfit

class EssentialSelector(BaseEstimator, TransformerMixin):

    def __init__(self, limit = np.inf):
        self.limit = limit

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        for i in range(num_diag):
            diag = X[i]
            idx_ess = diag[:,1] == self.limit
            Xfit.append(np.reshape(diag[:,0][idx_ess],[-1,1]))
        return Xfit










#############################################
# Finite Vectorization methods ##############
#############################################

class PersistenceImage(BaseEstimator, TransformerMixin):

    def __init__(self, kernel = lambda x, y: x[0]*y[0] + x[1]*y[1], weight = lambda x: 1,
                       resolution = [20,20], im_range = [np.nan, np.nan, np.nan, np.nan]):
        self.kernel, self.weight = kernel, weight
        self.resolution, self.range = resolution, im_range

    def fit(self, X, y = None):
        if np.isnan(self.range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.range = [mx, Mx, my, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True and isinstance(self.kernel, np.ndarray) == False:

                Xfit.append(np.array(persistence_image(diagram, self.range[0], self.range[1], self.resolution[0], self.range[2], self.range[3], self.resolution[1], self.kernel, self.weight)).flatten()[np.newaxis,:])

            else:

                w = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    w[j] = self.weight(diagram[j,:])

                if callable(self.kernel) == True:

                    x_values, y_values = np.linspace(self.range[0], self.range[1], self.resolution[0]), np.linspace(self.range[2], self.range[3], self.resolution[1])
                    Xs, Ys = np.meshgrid(x_values, y_values)
                    image = np.zeros(Xs.shape)
                    for j in range(self.resolution[1]):
                        for k in range(self.resolution[0]):
                            point = np.array([Xs[j,k], Ys[j,k]])
                            for l in range(num_pts_in_diag):
                                image[j,k] += w[l] * self.kernel(diagram[l,:], point)

                if isinstance(self.kernel, np.ndarray) == True:

                    image = convolve2d(    np.histogram2d(diagram[:,0], diagram[:,1], bins=self.resolution,
                                                          range=[[self.range[0],self.range[1]],[self.range[2],self.range[3]]], weights = w)[0],
                                           self.kernel, mode="same")

                Xfit.append(image.flatten()[np.newaxis,:])

        return np.concatenate(Xfit,0)

class Landscape(BaseEstimator, TransformerMixin):

    def __init__(self, num_landscapes = 5, resolution = 100, ls_range = [np.nan, np.nan]):
        self.num_landscapes, self.resolution, self.range = num_landscapes, resolution, ls_range

    def fit(self, X, y = None):
        if np.isnan(self.range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.range[0], self.range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(landscape(diagram, self.num_landscapes, self.range[0], self.range[1], self.resolution)).flatten()[np.newaxis,:])

            else:

                ls = np.zeros([self.num_landscapes, self.resolution])

                events = []
                for j in range(self.resolution):
                    events.append([])

                for j in range(num_pts_in_diag):
                    [px,py] = diagram[j,:]
                    min_idx = np.minimum(np.maximum(np.ceil((px          - self.range[0]) / step_x).astype(int), 0), self.resolution)
                    mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py          - self.range[0]) / step_x).astype(int), 0), self.resolution)

                    if min_idx < self.resolution and max_idx > 0:

                        landscape_value = self.range[0] + min_idx * step_x - px
                        for k in range(min_idx, mid_idx):
                            events[k].append(landscape_value)
                            landscape_value += step_x

                        landscape_value = py - self.range[0] - mid_idx * step_x
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

    def __init__(self, power = 1, resolution = 100, sh_range = [np.nan, np.nan]):
        self.power, self.resolution, self.range = power, resolution, sh_range

    def fit(self, X, y = None):
        if np.isnan(self.range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.range[0], self.range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            sh, weights = np.zeros(self.resolution), np.zeros(num_pts_in_diag)

            for j in range(num_pts_in_diag):
                [px,py] = diagram[j,:]
                weights[j] = np.power(np.abs(py-px), self.power)
            total_weight = np.sum(weights)

            for j in range(num_pts_in_diag):

                [px,py] = diagram[j,:]
                weight  = weights[j] / total_weight
                min_idx = np.minimum(np.maximum(np.ceil((px          - self.range[0]) / step_x).astype(int), 0), self.resolution)
                mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.range[0]) / step_x).astype(int), 0), self.resolution)
                max_idx = np.minimum(np.maximum(np.ceil((py          - self.range[0]) / step_x).astype(int), 0), self.resolution)

                if min_idx < self.resolution and max_idx > 0:

                    silhouette_value = self.range[0] + min_idx * step_x - px
                    for k in range(min_idx, mid_idx):
                        sh[k] += weight * silhouette_value
                        silhouette_value += step_x

                    silhouette_value = py - self.range[0] - mid_idx * step_x
                    for k in range(mid_idx, max_idx):
                        sh[k] += weight * silhouette_value
                        silhouette_value -= step_x

            Xfit.append(np.reshape(np.sqrt(2) * sh,[1,-1]))

        return np.concatenate(Xfit,0)

class BettiCurve(BaseEstimator, TransformerMixin):

    def __init__(self, resolution = 100, bc_range = [np.nan, np.nan]):
        self.resolution, self.range = resolution, bc_range

    def fit(self, X, y = None):
        if np.isnan(self.range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.range[0], self.range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]
            bc =  np.zeros(self.resolution)

            for j in range(num_pts_in_diag):
                [px,py] = diagram[j,:]
                min_idx, max_idx = np.ceil((px - self.range[0]) / step_x).astype(int), np.ceil((py - self.range[0]) / step_x).astype(int)

                for k in range(min_idx, max_idx):
                    bc[k] += 1

            Xfit.append(np.reshape(bc,[1,-1]))

        return np.concatenate(Xfit,0)

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
            pers = np.matmul(diagram, np.array([[-1.0],[1.0]]))
            min_pers = np.minimum(pers,np.transpose(pers))
            distances = DistanceMetric.get_metric("chebyshev").pairwise(diagram)
            vect = np.flip(np.sort(np.triu(np.minimum(distances, min_pers)), axis = None), 0)
            dim = np.minimum(len(vect), self.threshold)
            Xfit[i, :dim] = vect[:dim]

        return Xfit








#############################################
# Essential Vectorization methods ###########
#############################################

class EssentialDiagramVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        for i in range(num_diag):
            Xfit.append(np.reshape(self.vectorizer(X[i]),[1,-1]))
        return np.concatenate(Xfit,0)












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

    def __init__(self, N = 10, gaussian_bandwidth = 1.0):
        self.N = N
        self.gaussian_bandwidth = gaussian_bandwidth

    def fit(self, X, y = None):

        if USE_CYTHON == False:

            num_diag = len(X)
            angles = np.linspace(-np.pi/2, np.pi/2, self.N+1)
            self.step_angle = angles[1] - angles[0]
            self.thetas = np.concatenate([np.cos(angles[:-1])[np.newaxis,:], np.sin(angles[:-1])[np.newaxis,:]], 0)

            self.proj, self.proj_delta = [], []
            for i in range(num_diag):

                diagram = X[i]
                diag_thetas, list_proj = np.tensordot(diagram, self.thetas, 1), []
                for j in range(self.N):
                    list_proj.append( list(np.sort(diag_thetas[:,j])) )
                self.proj.append(list_proj)

                diagonal_diagram = np.tensordot(diagram, np.array([[0.5,0.5],[0.5,0.5]]), 1)
                diag_thetas, list_proj_delta = np.tensordot(diagonal_diagram, self.thetas, 1), []
                for j in range(self.N):
                    list_proj_delta.append( list(np.sort(diag_thetas[:,j])) )
                self.proj_delta.append(list_proj_delta)

        else:

            self.diagrams = X

        return self

    def transform(self, X):

        if USE_CYTHON == False:

            num_diag2 = len(X)
            proj, proj_delta = [], []
            for i in range(num_diag2):

                diagram = X[i]
                diag_thetas, list_proj = np.tensordot(diagram, self.thetas, 1), []
                for j in range(self.N):
                    list_proj.append( list(np.sort(diag_thetas[:,j])) )
                proj.append(list_proj)

                diagonal_diagram = np.tensordot(diagram, np.array([[0.5,0.5],[0.5,0.5]]), 1)
                diag_thetas, list_proj_delta = np.tensordot(diagonal_diagram, self.thetas, 1), []
                for j in range(self.N):
                    list_proj_delta.append( list(np.sort(diag_thetas[:,j])) )
                proj_delta.append(list_proj_delta)

            num_diag1 = len(self.proj)
            Xfit = np.zeros( [num_diag1, num_diag2] )
            for i in range(num_diag1):
                for j in range(num_diag2):

                    L1, L2 = [], []
                    for k in range(self.N):
                        lik, likd, ljk, ljkd = list(self.proj[i][k]), list(self.proj_delta[i][k]), list(proj[j][k]), list(proj_delta[j][k])
                        L1.append( np.array(mergeSorted(lik, ljkd))[:,np.newaxis] )
                        L2.append( np.array(mergeSorted(ljk, likd))[:,np.newaxis] )
                    L1, L2 = np.concatenate(L1,1), np.concatenate(L2,1)

                    Xfit[i,j] = np.sum(self.step_angle*np.sum(np.abs(L1-L2),0)/np.pi)

            Xfit =  np.exp(-Xfit/(2*self.gaussian_bandwidth*self.gaussian_bandwidth))

        else:
            Xfit = np.array(sliced_wasserstein_matrix(self.diagrams, X, self.gaussian_bandwidth, self.N))

        return Xfit

class PersistenceWeightedGaussian(BaseEstimator, TransformerMixin):

    def __init__(self, kernel = lambda x, y: x[0]*y[0] + x[1]*y[1], weight = lambda x: 1, use_pss = False):

        self.kernel             = kernel
        self.weight             = weight
        self.use_pss            = use_pss

    def fit(self, X, y = None):

        if self.use_pss == True:
            for i in range(len(X)):
                op_X = np.tensordot(X[i],np.array([[0.0,1.0],[1.0,0.0]]), 1)
                X[i] = np.concatenate([X[i],op_X], 0)

        self.diagrams = X

        return self

    def transform(self, X):

        if self.use_pss == True:
            for i in range(len(X)):
                op_X = np.tensordot(X[i],np.array([[0.0,1.0],[1.0,0.0]]), 1)
                X[i] = np.concatenate([X[i],op_X], 0)

        if USE_CYTHON == True:

            Xfit = np.array(persistence_weighted_gaussian_matrix(self.diagrams, X, self.kernel, self.weight))

        else:

            self.w, w = [], []

            for i in range(len(X)):
                num_pts_in_diag = X[i].shape[0]
                we = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    we[j] = self.weight(X[i][j,:])
                w.append(we)

            for i in range(len(self.diagrams)):
                num_pts_in_diag = self.diagrams[i].shape[0]
                we = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    we[j] = self.weight(self.diagrams[i][j,:])
                self.w.append(we)

            num_diag1, num_diag2 = len(self.w), len(X)
            Xfit = np.zeros([num_diag1, num_diag2])

            for i in range(num_diag1):
                num_pts1 = self.diagrams[i].shape[0]
                for j in range(num_diag2):
                    num_pts2 = X[j].shape[0]

                    kernel_matrix = np.zeros( [num_pts1, num_pts2] )
                    for k in range(num_pts1):
                        for l in range(num_pts2):
                            kernel_matrix[k,l] = self.kernel( self.diagrams[i][k,:], X[j][l,:]  )

                    Xfit[i,j] = np.tensordot(w[j], np.tensordot(self.w[i], kernel_matrix, 1), 1)

        return Xfit

class PersistenceScaleSpace(BaseEstimator, TransformerMixin):

    def __init__(self, gaussian_bandwidth = 1.0):
        self.PWG = PersistenceWeightedGaussian(kernel = lambda x, y: np.exp( (-(x[0]-y[0])*(x[0]-y[0]) -(x[1]-y[1])*(x[1]-y[1]))/(2*gaussian_bandwidth) ), weight = lambda x: 1 if x[1] >= x[0] else -1, use_pss = True)

    def fit(self, X, y = None):
        self.PWG.fit(X,y)
        return self

    def transform(self, X):
        return self.PWG.transform(X)






#############################################
# Metrics ###################################
#############################################

def compute_wass_matrix(diags1, diags2, p = 1, delta = 0.001):

    num_diag1 = len(diags1)

    if diags1 == diags2:
        matrix = np.zeros((num_diag1, num_diag1))

        if USE_CYTHON == True:
            if np.isinf(p):
                for i in range(num_diag1):
                    for j in range(i+1, num_diag1):
                        matrix[i,j] = bottleneck(diags1[i], diags1[j], delta)
                        matrix[j,i] = matrix[i,j]
            else:
                for i in range(num_diag1):
                    for j in range(i+1, num_diag1):
                        matrix[i,j] = wasserstein(diags1[i], diags1[j], p, delta)
                        matrix[j,i] = matrix[i,j]
        else:
            print("Cython required---returning null matrix")

    else:
        num_diag2 = len(diags2)
        matrix = np.zeros((num_diag1, num_diag2))

        if USE_CYTHON == True:
            if np.isinf(p):
                for i in range(num_diag1):
                    for j in range(num_diag2):
                        matrix[i,j] = bottleneck(diags1[i], diags2[j], delta)
            else:
                for i in range(num_diag1):
                    for j in range(num_diag2):
                        matrix[i,j] = wasserstein(diags1[i], diags2[j], p, delta)
        else:
            print("Cython required---returning null matrix")

    return matrix

class WassersteinDistance(BaseEstimator, TransformerMixin):

    def __init__(self, wasserstein_parameter = 1, delta = 0.001):
        self.wasserstein_parameter = wasserstein_parameter
        self.delta = delta

    def fit(self, X, y = None):
        self.diagrams = X
        return self

    def transform(self, X):
        return compute_wass_matrix(X, self.diagrams, self.wasserstein_parameter, self.delta)
