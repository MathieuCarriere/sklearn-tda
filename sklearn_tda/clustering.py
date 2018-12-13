"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
import itertools
from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.cluster         import DBSCAN
from sklearn.metrics         import pairwise_distances
from sklearn.neighbors       import radius_neighbors_graph, kneighbors_graph
from scipy.spatial.distance  import directed_hausdorff
from scipy.sparse            import csgraph

try:
    import gudhi as gd
    USE_GUDHI = True

except ImportError:
    USE_GUDHI = False
    print("Gudhi not found--GraphInducedComplex not available")

#############################################
# Clustering ################################
#############################################

class NNClustering(BaseEstimator, TransformerMixin):

    def __init__(self, radius, metric = "euclidean"):
        self.radius, self.metric = radius, metric

    def fit_predict(self, X):
        if type(self.radius) is int:
            adj = kneighbors_graph(X, n_neighbors = self.radius, metric = self.metric)
        else:
            adj = radius_neighbors_graph(X, radius = self.radius, metric = self.metric)
        _, clusters = csgraph.connected_components(adj)
        return clusters


class MapperComplex(BaseEstimator, TransformerMixin):

    def __init__(self, filters = np.array([[0]]), colors = np.array([[0]]), resolutions = -1, gains = .3, clustering = DBSCAN(), mask = 0, verbose = False,
                       beta = 0., C = 100, N = 100):
        self.filters, self.resolutions, self.gains, self.colors, self.clustering = filters, resolutions, gains, colors, clustering
        self.mask, self.verbose = mask, verbose
        self.beta, self.C, self.N = beta, C, N

    def get_optimal_parameters_for_hierarchical_clustering(self, X):

        if self.filters.shape[0] == 1:
            filters = X[:,self.filters.flatten()]
        else:
            filters = self.filters

        num_pts, num_filt, delta = X.shape[0], filters.shape[1], 0
        m = int(  num_pts / np.exp((1+self.beta) * np.log(np.log(num_pts)/np.log(self.C)))  )
        for _ in range(self.N):
            subpop = np.random.choice(num_pts, size = m, replace = False)
            d, _, _ = directed_hausdorff(X, X[subpop,:])
            delta += d/self.N

        pairwise = pairwise_distances(X)
        pairs = np.argwhere(pairwise <= delta)
        num_pairs = pairs.shape[0]
        res = []
        for f in range(num_filt):
            F = filters[:,f]
            minf, maxf = np.min(F), np.max(F)
            resf = 0
            for p in range(num_pairs):
                resf = max(resf, abs(F[pairs[p,0]] - F[pairs[p,1]]))
            res.append(int((maxf-minf)/resf))

        return delta, res


    def fit(self, X, y = None):

        if self.filters.shape[0] == 1:
            filters = X[:, self.filters.flatten()]
        else:
            filters = self.filters

        if self.colors.shape[0] == 1:
            colors = X[:, self.colors.flatten()]
        else:
            colors = self.colors

        if isinstance(self.gains, float):
            gains = self.gains * np.ones([filters.shape[1]])
        else:
            gains = self.gains

        if self.resolutions == -1:
            delta, resolutions = self.get_optimal_parameters_for_hierarchical_clustering(X)
            clustering = NNClustering(radius = delta)
        else:
            resolutions = self.resolutions
            clustering  = self.clustering

        self.st_, self.graph_ = gd.SimplexTree(), []
        self.clus_colors, self.clus_size = dict(), dict()

        num_filters, num_colors = filters.shape[1], colors.shape[1]
        interval_inds, intersec_inds = np.empty(filters.shape), np.empty(filters.shape)
        for i in range(num_filters):
            f, r, g = filters[:,i], resolutions[i], gains[i]
            min_f, max_f = np.min(f), np.max(f)
            epsilon = pow(10, np.log10(abs(max_f)) - 5)
            interval_endpoints, l = np.linspace(min_f - epsilon, max_f + epsilon, num = r+1, retstep = True)
            intersec_endpoints = []
            for j in range(1, len(interval_endpoints)-1):
                intersec_endpoints.append(interval_endpoints[j] - g*l / (2 - 2*g))
                intersec_endpoints.append(interval_endpoints[j] + g*l / (2 - 2*g))
            interval_inds[:,i] = np.digitize(f, interval_endpoints)
            intersec_inds[:,i] = 0.5 * (np.digitize(f, intersec_endpoints) + 1)
            if self.verbose:
                print(interval_inds[:,i])
                print(intersec_inds[:,i])

        num_pts = filters.shape[0]
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

        clus_base = 0
        for preimage in binned_data:

            idxs = np.array(binned_data[preimage])
            clusters = clustering.fit_predict(X[idxs,:])

            if self.verbose:
                print("clusters in preimage " + str(preimage) + " = " + str(clusters))

            num_clus_pre = np.max(clusters) + 1
            for i in range(num_clus_pre):
                subpopulation = idxs[clusters == i]
                color_vals = np.mean(colors[subpopulation,:], axis=0)
                self.clus_colors[clus_base + i] = color_vals
                self.clus_size[clus_base + i] = len(subpopulation)

            for i in range(clusters.shape[0]):
                if clusters[i] != -1:
                    cover[idxs[i]].append(clus_base + clusters[i])

            clus_base += np.max(clusters) + 1

        for i in range(num_pts):
            self.st_.insert(cover[i], filtration = -3)

        for simplex in self.st_.get_skeleton(2):
            if len(simplex[0]) > 1:
                idx1, idx2 = simplex[0][0], simplex[0][1]
                if self.mask <= self.clus_size[idx1] and self.mask <= self.clus_size[idx2]:
                    self.graph_.append([simplex[0]])
            else:
                clus_idx = simplex[0][0]
                if self.mask <= self.clus_size[clus_idx]:
                    self.graph_.append([simplex[0], self.clus_colors[clus_idx], self.clus_size[clus_idx]])

        return self

    def persistence_diagram(self):
        list_dgm = []
        num_cols = self.clus_colors[list(self.clus_colors.keys())[0]].shape[0]
        for c in range(num_cols):
            col_vals = []
            for key, elem in self.clus_colors.items():
                col_vals.append(elem[c])
            st = gd.SimplexTree()
            list_simplices, list_vertices = self.st_.get_skeleton(1), self.st_.get_skeleton(0)
            for simplex in list_simplices:
                st.insert(simplex[0] + [-2], filtration = -3)
            min_val, max_val = min(col_vals), max(col_vals)
            for vertex in list_vertices:
                if st.find(vertex[0]):
                    st.assign_filtration(vertex[0],        filtration = -2 + (col_vals[vertex[0][0]]-min_val)/(max_val-min_val))
                    st.assign_filtration(vertex[0] + [-2], filtration =  2 - (col_vals[vertex[0][0]]-min_val)/(max_val-min_val))
            st.make_filtration_non_decreasing()
            dgm = st.persistence()
            for point in range(len(dgm)):
                b,d = dgm[point][1][0], dgm[point][1][1]
                b,d = min_val+(2-abs(b))*(max_val-min_val), min_val+(2-abs(d))*(max_val-min_val)
                dgm[point] = tuple([dgm[point][0], tuple([b,d])])
            list_dgm.append(dgm)
        return list_dgm

class GraphInducedComplex(BaseEstimator, TransformerMixin):

    def __init__(self, graph=-1, graph_subsampling=100, graph_subsampling_power=0.001, graph_subsampling_constant=10,
                       cover_type="functional", filter=0, resolution=-1, gain=0.33, Voronoi_subsampling=1000,
                       mask=0, color=0, verbose=False, input="point cloud"):

        if USE_GUDHI == False:
            raise ImportError("Error: Gudhi not imported")

        self.cc = gd.CoverComplex()
        self.cc.set_type("GIC")
        self.cc.set_mask(mask)
        self.cc.set_verbose(verbose)
        self.graph, self.graph_subsampling, self.graph_subsampling_constant, self.graph_subsampling_power = graph, graph_subsampling, graph_subsampling_constant, graph_subsampling_power
        self.cover_type, self.filter, self.resolution, self.gain, self.Voronoi_subsampling = cover_type, filter, resolution, gain, Voronoi_subsampling
        self.color, self.input = color, input

    def fit(self, X, y = None):

        # Read input
        if self.input == "point cloud":
            self.cc.set_point_cloud_from_range(X)
        elif self.input == "distance matrix":
            self.cc.set_distances_from_range(X)

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
