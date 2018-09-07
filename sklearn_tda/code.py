"""
@author: Mathieu Carriere
All rights reserved
"""

import sys
import numpy as np
import itertools
import matplotlib.cm as cm

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors     import DistanceMetric
from sklearn.cluster       import DBSCAN

try:
    from .vectors import *
    from .kernels import *
    from .hera_wasserstein import *
    from .hera_bottleneck import *
    USE_CYTHON = True

except ImportError:
    USE_CYTHON = False
    print("Cython not found--WassersteinDistance not available")

try:
    import gudhi as gd
    USE_GUDHI = True

except ImportError:
    USE_GUDHI = False
    print("Gudhi not found--GraphInducedComplex not available")

#############################################
# Clustering ################################
#############################################

class MapperComplex(BaseEstimator, TransformerMixin):

    def __init__(self, filters, resolutions, gains, color, clustering = DBSCAN(), epsilon = 1e-5, verbose = False):
        self.filters, self.resolutions, self.gains, self.color, self.clustering = filters, resolutions, gains, color, clustering
        self.epsilon, self.verbose = epsilon, verbose

    def fit(self, X, y = None):

        num_filters = self.filters.shape[1]
        interval_inds, intersec_inds = np.empty(self.filters.shape), np.empty(self.filters.shape)
        for i in range(num_filters):
            f, r, g = self.filters[:,i], self.resolutions[i], self.gains[i]
            min_f, max_f = np.min(f), np.max(f)
            interval_endpoints, l = np.linspace(min_f - self.epsilon, max_f + self.epsilon, num = r+1, retstep = True)
            intersec_endpoints = []
            for j in range(1, len(interval_endpoints)-1):
                intersec_endpoints.append(interval_endpoints[j] - g*l / (2 - 2*g))
                intersec_endpoints.append(interval_endpoints[j] + g*l / (2 - 2*g))
            interval_inds[:,i] = np.digitize(f, interval_endpoints)
            intersec_inds[:,i] = 0.5 * (np.digitize(f, intersec_endpoints) + 1)

        num_pts = self.filters.shape[0]
        binned_data = dict()
        for i in range(num_pts):
            list_preimage = []
            for j in range(num_filters):
                a, b = interval_inds[i,j], intersec_inds[i,j]
                list_preimage.append([a])
                if b == a:
                    list_preimage[j].append(a+1)
                if b == a-1:
                    list_preimage[j].append(a-1)
            list_preimage = list(itertools.product(*list_preimage))
            for pre_idx in list_preimage:
                if pre_idx in binned_data:
                    binned_data[pre_idx].append(i)
                else:
                    binned_data[pre_idx] = [i]

        if self.verbose:
            print(binned_data)

        cover = []
        for i in range(num_pts):
            cover.append([])

        clus_base, clus_color, clus_size = 0, dict(), dict()
        for preimage in binned_data:

            idxs = np.array(binned_data[preimage])
            clusters = self.clustering.fit_predict(X[idxs,:])

            if self.verbose:
                print("clusters in preimage " + str(preimage) + " = " + str(clusters))

            num_clus_pre = np.max(clusters) + 1
            for i in range(num_clus_pre):
                subpopulation = idxs[clusters == i]
                color_val = np.mean(self.color[subpopulation])
                clus_color[clus_base + i] = color_val
                clus_size[clus_base + i] = len(subpopulation)

            for i in range(clusters.shape[0]):
                if clusters[i] != -1:
                    cover[idxs[i]].append(clus_base + clusters[i])

            clus_base += np.max(clusters) + 1

        self.st_ = gd.SimplexTree()
        for i in range(num_pts):
            num_clus_i = len(cover[i])
            for j in range(num_clus_i):
                self.st_.insert([cover[i][j]])
            self.st_.insert(cover[i])

        self.graph_ = []
        for simplex in self.st_.get_skeleton(2):
            print(simplex)
            if len(simplex[0]) > 1:
                self.graph_.append([simplex[0]])
            else:
                clus_idx = simplex[0][0]
                self.graph_.append([simplex[0], clus_color[clus_idx], clus_size[clus_idx]])

        return self


class GraphInducedComplex(BaseEstimator, TransformerMixin):

    def __init__(self, graph = -1, graph_subsampling = 100, graph_subsampling_power = 0.001, graph_subsampling_constant = 10,
                       cover_type = "functional", filter = 0, resolution = -1, gain = 0.33, Voronoi_subsampling = 1000,
                       mask = 0, color = 0, verbose = False):

        if USE_GUDHI == False:
            raise ImportError("Error: Gudhi not imported")

        self.cc = gd.CoverComplex()
        self.cc.set_type("GIC")
        self.cc.set_mask(mask)
        self.cc.set_verbose(verbose)
        self.graph, self.graph_subsampling, self.graph_subsampling_constant, self.graph_subsampling_power = graph, graph_subsampling, graph_subsampling_constant, graph_subsampling_power
        self.cover_type, self.filter, self.resolution, self.gain, self.Voronoi_subsampling = cover_type, filter, resolution, gain, Voronoi_subsampling
        self.color = color

    def fit(self, X, y = None):

        # Read input
        self.cc.set_point_cloud_from_range(X)

        # Set color function
        if type(self.color) is int:
            self.cc.set_color_from_coordinate(self.color)
        if type(self.color) is np.ndarray:
            self.cc.set_color_from_range(self.color)

        # Set underlying neighborhood graph for connected components
        if self.graph == -1:
            self.cc.set_subsampling(self.graph_subsampling_constant, self.graph_subsampling_power)
            self.cc.set_graph_from_automatic_rips(self.graph_subsampling)
        else:
            self.cc.set_graph_from_rips(self.graph)

        # Set cover of point cloud
        if self.cover_type == "functional":
            ###### Function values
            if type(self.filter) is int:
                self.cc.set_function_from_coordinate(self.filter)
            if type(self.filter) is np.ndarray:
                self.cc.set_function_from_range(self.filter)
            ###### Gain
            self.cc.set_gain(self.gain)
            ###### Resolution
            if self.resolution == -1:
                self.cc.set_automatic_resolution()
            else:
                if type(self.resolution) is int:
                    self.cc.set_resolution_with_interval_number(self.resolution)
                else:
                    self.cc.set_resolution_with_interval_length(self.resolution)
            ###### Cover computation
            self.cc.set_cover_from_function()
        if self.cover_type == "Voronoi":
            self.cc.set_cover_from_Voronoi(self.Voronoi_subsampling)

        # Compute simplices
        self.cc.find_simplices()
        self.cc.create_simplex_tree()

        return self

    def print_result(self, output_type = "txt"):
        if output_type == "txt":
            self.cc.write_info()
        if output_type == "dot":
            self.cc.plot_dot()
        if output_type == "off":
            self.cc.plot_off()

    def compute_p_value(self, bootstrap = 10):
        self.cc.compute_distribution(bootstrap)
        return self.cc.compute_p_value()

    def compute_confidence_level_from_distance(self, bootstrap = 10, distance = 1.0):
        self.cc.compute_distribution(bootstrap)
        return self.cc.compute_confidence_level_from_distance(distance)

    def compute_distance_from_confidence_level(self, bootstrap = 10, alpha = 0.1):
        self.cc.compute_distribution(bootstrap)
        return self.cc.compute_distance_from_confidence_level(alpha)

    def subpopulation(self, node_index = 0):
        return self.cc.subpopulation(node_index)

    def persistence_diagram(self):
        return self.cc.compute_PD()










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
        self.num_pts    = num_pts
        self.threshold  = threshold
        self.use        = use

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
                    sorted_diag = thresh_diag[sort_index[:min(self.num_pts, diag.shape[0])],:]
                    Xfit.append(sorted_diag)
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






#############################################
# Metrics ###################################
#############################################

def compute_wass_matrix(diags1, diags2, p = 1, delta = 0.001):

    num_diag1 = len(diags1)

    if np.array_equal(np.concatenate(diags1,0), np.concatenate(diags2,0)) == True:
        matrix = np.zeros((num_diag1, num_diag1))

        if USE_CYTHON == True:
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

        if USE_CYTHON == True:
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
