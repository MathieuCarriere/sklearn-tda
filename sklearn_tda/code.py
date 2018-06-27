# -*- coding: utf-8 -*-
"""
Created on

@author: Mathieu Carrière
All rights reserved
"""

import numpy             as np
from   scipy.signal      import convolve2d

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import *

try:
    import gudhi as gd
    USE_GUDHI = True
    print("Gudhi found")
except ImportError:
    USE_GUDHI = False
    print("Gudhi not found")

try:
    from .vectors import *
    from .kernels import *
    from .hera_wasserstein import *
    from .hera_bottleneck import *
    USE_CYTHON = True
    print("Cython found")
except ImportError:
    USE_CYTHON = False
    print("Cython not found")

#############################################
# Preprocessing #############################
#############################################

class BirthPersistenceTransform(BaseEstimator, TransformerMixin):

    def __init__(self, scaler = StandardScaler()):
        return None

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return np.tensordot(X, np.array([[1.0, -1.0],[0.0, 1.0]]), 1)


class DiagramPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, scaler = StandardScaler()):
        self.scaler = scaler

    def fit(self, X, y = None):
        if len(X) == 1:
            P = X[0]
        else:
            P = np.concatenate(X,0)
        self.scaler.fit(P)
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        for i in range(num_diag):
            diag = X[i]
            if diag.shape[0] > 0:
                diag = self.scaler.transform(diag)
                Xfit.append(diag)
        return Xfit

class ProminentPoints(BaseEstimator, TransformerMixin):

    def __init__(self, num_pts = 10):
        self.num_pts = num_pts

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        for i in range(num_diag):
            diag         = X[i]
            sort_index   = np.flip(np.argsort(np.matmul(diag, [-1.0, 1.0])),0)
            sorted_diag  = diag[sort_index[:min(self.num_pts,diag.shape[0])],:]
            Xfit.append(sorted_diag)
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

class FiniteDiagramVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, name = "PersistenceImage",
                       kernel = "rbf", gaussian_bandwidth = 1.0, polynomial_bias = 1.0, polynomial_power = 1.0, weight = "const",
                       resolution_x = 20, resolution_y = 20, min_x = np.nan, max_x = np.nan, min_y = np.nan, max_y = np.nan,
                       num_landscapes = 5):

        self.name                = name

        self.kernel              = kernel
        self.gaussian_bandwidth  = gaussian_bandwidth
        self.polynomial_bias     = polynomial_bias
        self.polynomial_power    = polynomial_power
        self.weight              = weight

        self.resolution_x        = resolution_x
        self.resolution_y        = resolution_y
        self.min_x               = min_x
        self.max_x               = max_x
        self.min_y               = min_y
        self.max_y               = max_y

        self.num_landscapes      = num_landscapes

    def fit(self, X, y = None):
        if self.name == "PersistenceImage":
            self.vectorizer = PersistenceImage(self.kernel, self.gaussian_bandwidth, self.polynomial_bias, self.polynomial_power, self.weight,
                                               self.resolution_x, self.resolution_y, self.min_x, self.max_x, self.min_y, self.max_y)
        if self.name == "Landscape":
            self.vectorizer = Landscape(self.num_landscapes, self.resolution_x, self.min_x, self.max_x)

        if self.name == "BettiCurve":
            self.vectorizer = BettiCurve(self.resolution_x, self.min_x, self.max_x)

        return self.vectorizer.fit(X, y)

class PersistenceImage(BaseEstimator, TransformerMixin):

    def __init__(self, kernel = "rbf", gaussian_bandwidth = 1.0, polynomial_bias = 1.0, polynomial_power = 1.0, weight = lambda x: 1,
                       resolution_x = 20, resolution_y = 20, min_x = np.nan, max_x = np.nan, min_y = np.nan, max_y = np.nan):
        self.kernel, self.gaussian_bandwidth, self.polynomial_bias, self.polynomial_power, self.weight = kernel, gaussian_bandwidth, polynomial_bias, polynomial_power, weight
        self.resolution_x, self.resolution_y, self.min_x, self.max_x, self.min_y, self.max_y = resolution_x, resolution_y, min_x, max_x, min_y, max_y

    def fit(self, X, y = None):
        if np.isnan(self.min_x) == True:
            pre = DiagramPreprocessor(MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.min_x, self.max_x, self.min_y, self.max_y = mx, Mx, my, My
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_GUDHI == True and isinstance(self.kernel, str) == True and isinstance(self.weight, str) == True:

                Xfit.append(np.array(gd.persistence_image(diagram, min_x = self.min_x, max_x = self.max_x, res_x = self.resolution_x, min_y = self.min_y, max_y = self.max_y, res_y = self.resolution_y, weight = "linear", sigma = 1.0)).flatten()[np.newaxis,:])

            else:

                if USE_CYTHON == True and isinstance(self.kernel, str) == True and isinstance(self.weight, str) == True:

                    Xfit.append(np.array(persistence_image(diagram, self.min_x, self.max_x, self.resolution_x, self.min_y, self.max_y, self.resolution_y, self.kernel, self.weight, self.gaussian_bandwidth, self.polynomial_bias, self.polynomial_power)).flatten()[np.newaxis,:])

                else:

                    if callable(self.weight) == False:
                        raise ValueError("Weight function needs to be defined in Python")

                    w = np.ones(num_pts_in_diag)
                    for j in range(num_pts_in_diag):
                        w[j] = self.weight(diagram[j,:])

                    if isinstance(self.kernel, str) == True and self.kernel == "rbf":

                        x_values, y_values = np.linspace(self.min_x, self.max_x, self.resolution_x), np.linspace(self.min_y, self.max_y, self.resolution_y)
                        Xs, Ys = np.tile((diagram[:,0][:,np.newaxis,np.newaxis]-x_values[np.newaxis,np.newaxis,:]),[1,self.resolution_y,1]), np.tile(diagram[:,1][:,np.newaxis,np.newaxis]-y_values[np.newaxis,:,np.newaxis],[1,1,self.resolution_x])
                        image = np.tensordot(w, np.exp((-np.square(Xs)-np.square(Ys))/(2*self.gaussian_bandwidth*self.gaussian_bandwidth))/(self.gaussian_bandwidth*np.sqrt(2*np.pi)), 1)

                    if isinstance(self.kernel, str) == True and self.kernel == "poly":

                        x_values, y_values = np.linspace(self.min_x, self.max_x, self.resolution_x), np.linspace(self.min_y, self.max_y, self.resolution_y)
                        Xs, Ys = np.meshgrid(x_values, y_values)
                        image = np.tensordot(w, np.power(np.tensordot(  diagram, np.concatenate([Xs[np.newaxis,:],Ys[np.newaxis,:]],0), 1) + self.polynomial_bias, self.polynomial_power), 1)

                    if callable(self.kernel) == True:

                        x_values, y_values = np.linspace(self.min_x, self.max_x, self.resolution_x), np.linspace(self.min_y, self.max_y, self.resolution_y)
                        Xs, Ys = np.meshgrid(x_values, y_values)
                        image = np.zeros(Xs.shape)
                        for j in range(self.resolution_y):
                            for k in range(self.resolution_x):
                                point = np.array([Xs[j,k], Ys[j,k]])
                                for l in range(num_pts_in_diag):
                                    image[j,k] += w[l] * self.kernel(diagram[l,:], point)

                    if isinstance(self.kernel, np.ndarray) == True:

                        image = convolve2d(    np.histogram2d(diagram[:,0], diagram[:,1], bins=[self.resolution_x,self.resolution_y],
                                                              range=[[self.min_x,self.max_x],[self.min_y,self.max_y]], weights = w)[0],
                                               self.kernel, mode="same")

                    Xfit.append(image.flatten()[np.newaxis,:])

        return np.concatenate(Xfit,0)

class Landscape(BaseEstimator, TransformerMixin):

    def __init__(self, num_landscapes = 5, resolution_x = 100, min_x = np.nan, max_x = np.nan):
        self.num_landscapes, self.resolution_x, self.min_x, self.max_x = num_landscapes, resolution_x, min_x, max_x

    def fit(self, X, y = None):
        if np.isnan(self.min_x) == True:
            pre = DiagramPreprocessor(MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.min_x, self.max_x = mx, My
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.min_x, self.max_x, self.resolution_x)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_GUDHI == False:

                if USE_CYTHON == True:

                    Xfit.append(np.array(landscape(diagram, self.num_landscapes, self.min_x, self.max_x, self.resolution_x)).flatten()[np.newaxis,:])

                else:

                    ls = np.zeros([self.num_landscapes, self.resolution_x])

                    events = []
                    for j in range(self.resolution_x):
                        events.append([])

                    for j in range(num_pts_in_diag):
                        [px,py] = diagram[j,:]
                        min_idx = np.minimum(np.maximum(np.ceil((px          - self.min_x) / step_x).astype(int), 0), self.resolution_x)
                        mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.min_x) / step_x).astype(int), 0), self.resolution_x)
                        max_idx = np.minimum(np.maximum(np.ceil((py          - self.min_x) / step_x).astype(int), 0), self.resolution_x)

                        if min_idx < self.resolution_x and max_idx > 0:

                            landscape_value = self.min_x + min_idx * step_x - px
                            for k in range(min_idx, mid_idx):
                                events[k].append(landscape_value)
                                landscape_value += step_x

                            landscape_value = py - self.min_x - mid_idx * step_x
                            for k in range(mid_idx, max_idx):
                                events[k].append(landscape_value)
                                landscape_value -= step_x

                    for j in range(self.resolution_x):
                        events[j].sort(reverse = True)
                        for k in range( min(self.num_landscapes, len(events[j])) ):
                            ls[k,j] = events[j][k]

                    Xfit.append(np.sqrt(2)*np.reshape(ls,[1,-1]))

            else:

                Xfit.append(np.array(gd.landscape(diagram, self.num_landscapes, self.min_x, self.max_x, self.resolution_x)).flatten()[np.newaxis,:])

        return np.concatenate(Xfit,0)

class BettiCurve(BaseEstimator, TransformerMixin):

    def __init__(self, resolution_x = 100, min_x = np.nan, max_x = np.nan):
        self.resolution_x, self.min_x, self.max_x = resolution_x, min_x, max_x

    def fit(self, X, y = None):
        if np.isnan(self.min_x) == True:
            pre = DiagramPreprocessor(MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.min_x, self.max_x = mx, My
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.min_x, self.max_x, self.resolution_x)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]
            bc =  np.zeros(self.resolution_x)

            for j in range(num_pts_in_diag):
                [px,py] = diagram[j,:]
                min_idx, max_idx = np.ceil((px - self.min_x) / step_x).astype(int), np.ceil((py - self.min_x) / step_x).astype(int)

                for k in range(min_idx, max_idx):
                    bc[k] += 1

            Xfit.append(np.reshape(bc,[1,-1]))

        return np.concatenate(Xfit,0)










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

class DiagramKernelizer(BaseEstimator, TransformerMixin):

    def __init__(self, name = "SlicedWasserstein",
                       N = 10,
                       kernel = "rbf", gaussian_bandwidth = 1.0, polynomial_bias = 1.0, polynomial_power = 1.0, weight = lambda x: 1, use_pss = False):

        self.name               = name

        self.N                  = N

        self.kernel             = kernel
        self.gaussian_bandwidth = gaussian_bandwidth
        self.polynomial_bias    = polynomial_bias
        self.polynomial_power   = polynomial_power
        self.weight             = weight
        self.use_pss            = use_pss

    def fit(self, X, y = None):

        if(self.name == "SlicedWasserstein"):
            self.kernel = SlicedWasserstein(N=self.N, gaussian_bandwidth=self.gaussian_bandwidth)

        if(self.name == "PersistenceScaleSpace"):
            self.kernel = PersistenceWeightedGaussian(kernel = "rbf", gaussian_bandwidth = self.gaussian_bandwidth, weight = lambda x: 1 if x[1] >= x[0] else -1, use_pss = True)

        if(self.name == "PersistenceWeightedGaussian"):
            self.kernel = PersistenceWeightedGaussian(kernel = self.kernel, gaussian_bandwidth=self.gaussian_bandwidth, polynomial_bias = self.polynomial_bias,
                                                      polynomial_power = self.polynomial_power, weight = self.weight, use_pss = False)
        return self.kernel.fit(X, y)

    def transform(self, X):
        return self.kernel.transform(X)


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

        if USE_GUDHI == False and USE_CYTHON == False:

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

        if USE_GUDHI == False:

            if USE_CYTHON == False:

                if self.N == -1:
                    raise ValueError("Exact computation is not available in Python")

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

        else:

            Xfit = np.array(gd.sliced_wasserstein_matrix(self.diagrams, X, self.gaussian_bandwidth, self.N))

        return Xfit

class PersistenceWeightedGaussian(BaseEstimator, TransformerMixin):

    def __init__(self, kernel = "rbf", gaussian_bandwidth = 1.0, polynomial_bias = 1.0, polynomial_power = 1.0, weight = lambda x: 1, use_pss = False):

        self.kernel             = kernel
        self.gaussian_bandwidth = gaussian_bandwidth
        self.polynomial_bias    = polynomial_bias
        self.polynomial_power   = polynomial_power
        self.weight             = weight
        self.use_pss            = use_pss

    def fit(self, X, y= None):

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

        if USE_GUDHI == True and isinstance(self.kernel, str) == True and isinstance(self.weight, str) == True:

            #TODO: add gudhi code
            Xfit = np.array(gd.persistence_weighted_gaussian_matrix())

        else:

            if USE_CYTHON == True and isinstance(self.kernel, str) == True and isinstance(self.weight, str) == True:

                Xfit = np.array(persistence_weighted_gaussian_matrix(self.diagrams, X, self.kernel, self.weight, self.gaussian_bandwidth, self.polynomial_bias, self.polynomial_power))

            else:

                if callable(self.weight) == False:
                    raise ValueError("Weight function needs to be defined in Python")

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

                if isinstance(self.kernel, str) == True and self.kernel == "rbf":
                    for i in range(num_diag1):
                        for j in range(num_diag2):
                            d1x, d1y, d2x, d2y = self.diagrams[i][:,0][:,np.newaxis], self.diagrams[i][:,1][:,np.newaxis], X[j][:,0][np.newaxis,:], X[j][:,1][np.newaxis,:]
                            Xfit[i,j] = np.tensordot(w[j], np.tensordot(self.w[i], np.exp( -(np.square(d1x-d2x) + np.square(d1y-d2y)) / (2*self.gaussian_bandwidth*self.gaussian_bandwidth)) / (self.gaussian_bandwidth*np.sqrt(2*np.pi)), 1), 1)

                if isinstance(self.kernel, str) == True and self.kernel == "poly":
                    for i in range(num_diag1):
                        for j in range(num_diag2):
                            Xfit[i,j] = np.tensordot(w[j], np.tensordot(self.w[i], np.power(np.tensordot(self.diagrams[i], np.transpose(X[j]), 1) + self.polynomial_bias, self.polynomial_power), 1), 1)

                if callable(self.kernel) == True:
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
        num_diag2 = len(X2)
        matrix = np.zeros((num_diag1, num_diag2))

        if USE_CYTHON == True:
            if np.isinf(p):
                for i in range(num_diag1):
                    for j in range(num_diag2):
                        matrix[i,j] = bottleneck.bottleneck(diags1[i], diags2[j], delta)
            else:
                for i in range(num_diag1):
                    for j in range(num_diag2):
                        matrix[i,j] = wasserstein(diags1[i], diags2[j], p, delta)
        else:
            print("Cython required---returning null matrix")

    return matrix

class DiagramMetrizer(BaseEstimator, TransformerMixin):

    def __init__(self, wasserstein_parameter = 1, delta = 0.001):
        self.wasserstein_parameter = wasserstein_parameter
        self.delta = delta

    def fit(self, X, y = None):
        self.diagrams = X
        return self

    def transform(self, X):
        return compute_wass_matrix(X, self.diagrams, self.wasserstein_parameter, self.delta)
