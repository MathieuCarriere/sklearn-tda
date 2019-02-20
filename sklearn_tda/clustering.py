"""
@author: Mathieu Carriere
All rights reserved
"""

import numpy as np
import itertools

from .metrics                import WassersteinDistance
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

    def __init__(self, radius, metric="euclidean", input="point cloud"):
        self.radius_, self.metric_ = radius, metric

    def fit_predict(self, X):
        if type(self.radius_) is int:
            if input == "point cloud":
                adj = kneighbors_graph(X, n_neighbors=self.radius_, metric=self.metric_)
            if input == "distance matrix":
                adj = np.zeros(X.shape)
                idxs = np.argpartition(X, self.radius_, axis=1)[:, :self.radius_]
                for i in range(len(X)):
                    adj[i,idxs[i,:]] = np.ones(len(idxs[i]))                    
        else:
            if input == "point cloud":
                adj = radius_neighbors_graph(X, radius=self.radius_, metric=self.metric_)
            if input == "distance matrix":
                adj = np.where(X <= self.radius_, np.ones(X.shape), np.zeros(X.shape))
        _, clusters = csgraph.connected_components(adj)
        return clusters


class MapperComplex(BaseEstimator, TransformerMixin):

    def __init__(self, filters=np.array([[0]]), filter_bnds="auto", colors=np.array([[0]]), resolutions=-1, gains=.3, clustering=DBSCAN(), 
                       mask=0, beta=0., C=100, N=100, input="point cloud", verbose=False):
        self.filters_, self.filter_bnds_, self.resolutions_, self.gains_, self.colors_, self.clustering_ = filters, filter_bnds, resolutions, gains, colors, clustering
        self.mask_, self.verbose_ = mask, verbose
        self.input_, self.beta_, self.C_, self.N_ = input, beta, C, N

    def get_optimal_parameters_for_hierarchical_clustering(self, X):

        if self.filters_.shape[0] == 1 and self.input_ == "point cloud":
            filters = X[:,self.filters_.flatten()]
        else:
            filters = self.filters_

        num_pts, num_filt, delta = X.shape[0], filters.shape[1], 0
        m = int(  num_pts / np.exp((1+self.beta_) * np.log(np.log(num_pts)/np.log(self.C_)))  )
        for _ in range(self.N_):
            subpop = np.random.choice(num_pts, size=m, replace=False)
            if self.input_ == "point cloud":
                d, _, _ = directed_hausdorff(X, X[subpop,:])
            if self.input_ == "distance matrix":
                d = np.max(np.min(X[:,subpop], axis=1), axis=0)
            delta += d/self.N_

        if self.input_ == "point cloud":
            pairwise = pairwise_distances(X, metric="euclidean")
        if self.input_ == "distance matrix":
            pairwise = X
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


    def fit(self, X, y=None):

        if self.filters_.shape[0] == 1:
            if self.input_ == "point cloud":
                filters = X[:, self.filters_.flatten()]
            else:
                print("Cannot set filters as coordinates when input is a distance matrix---using eccentricity instead")
                filters = np.max(X, axis=1)[:,np.newaxis]
        else:
            filters = self.filters_

        if self.colors_.shape[0] == 1:
            if self.input_ == "point cloud":
                colors = X[:, self.colors_.flatten()]
            else:
                print("Cannot set colors as coordinates when input is a distance matrix---using null function instead")
                colors = np.zeros([X.shape[0],1])
        else:
            colors = self.colors_

        if isinstance(self.gains_, float):
            gains = self.gains_ * np.ones([filters.shape[1]])
        else:
            gains = self.gains_

        if self.resolutions_ == -1:
            delta, resolutions = self.get_optimal_parameters_for_hierarchical_clustering(X)
            clustering = NNClustering(radius=delta, input=self.input_)
        else:
            resolutions = self.resolutions_
            clustering  = self.clustering_

        self.st_, self.graph_ = gd.SimplexTree(), []
        self.clus_colors_, self.clus_size_, self.clus_name_, self.clus_subpop_ = dict(), dict(), dict(), dict()

        num_filters, num_colors = filters.shape[1], colors.shape[1]
        interval_inds, intersec_inds = np.empty(filters.shape), np.empty(filters.shape)
        for i in range(num_filters):
            f, r, g = filters[:,i], resolutions[i], gains[i]
            if self.filter_bnds_ == "auto":
                min_f, max_f = np.min(f), np.max(f)
                epsilon = pow(10, np.log10(abs(max_f)) - 5)
                interval_endpoints, l = np.linspace(min_f - epsilon, max_f + epsilon, num=r+1, retstep=True)
            else:
                min_f, max_f = self.filter_bnds_[i,0], self.filter_bnds_[i,1]
                interval_endpoints, l = np.linspace(min_f, max_f, num=r+1, retstep=True)
            intersec_endpoints = []
            for j in range(1, len(interval_endpoints)-1):
                intersec_endpoints.append(interval_endpoints[j] - g*l / (2 - 2*g))
                intersec_endpoints.append(interval_endpoints[j] + g*l / (2 - 2*g))
            interval_inds[:,i] = np.digitize(f, interval_endpoints)
            intersec_inds[:,i] = 0.5 * (np.digitize(f, intersec_endpoints) + 1)
            if self.verbose_:
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

        if self.verbose_:
            print(binned_data)

        cover = []
        for i in range(num_pts):
            cover.append([])

        clus_base = 0
        for preimage in binned_data:

            idxs = np.array(binned_data[preimage])
            if self.input_ == "point cloud":
                clusters = clustering.fit_predict(X[idxs,:])
            if self.input_ == "distance matrix":
                clusters = clustering.fit_predict(X[idxs,:][:,idxs])

            if self.verbose_:
                print("clusters in preimage " + str(preimage) + " = " + str(clusters))

            num_clus_pre = np.max(clusters) + 1
            for i in range(num_clus_pre):
                subpopulation = idxs[clusters == i]
                color_vals = np.mean(colors[subpopulation,:], axis=0)
                self.clus_colors_[clus_base + i] = color_vals
                self.clus_size_  [clus_base + i] = len(subpopulation)
                self.clus_name_  [clus_base + i] = preimage
                self.clus_subpop_[clus_base + i] = subpopulation

            for i in range(clusters.shape[0]):
                if clusters[i] != -1:
                    cover[idxs[i]].append(clus_base + clusters[i])

            clus_base += np.max(clusters) + 1

        for i in range(num_pts):
            self.st_.insert(cover[i], filtration=-3)

        for simplex in self.st_.get_skeleton(2):
            if len(simplex[0]) > 1:
                idx1, idx2 = simplex[0][0], simplex[0][1]
                if self.mask_ <= self.clus_size_[idx1] and self.mask_ <= self.clus_size_[idx2]:
                    self.graph_.append([simplex[0]])
            else:
                clus_idx = simplex[0][0]
                if self.mask_ <= self.clus_size_[clus_idx]:
                    self.graph_.append([simplex[0], self.clus_colors_[clus_idx], self.clus_size_[clus_idx], self.clus_name_[clus_idx]])

        return self

    def persistence_diagram(self):
        list_dgm = []
        num_cols = self.clus_colors_[list(self.clus_colors_.keys())[0]].shape[0]
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

    def compute_distribution(self, X, N=100):
        num_pts, distribution = len(X), []
        for bootstrap_id in range(N):
            if self.verbose_:
                print(str(bootstrap_id) + "th iteration")
            idxs = np.random.choice(num_pts, size=num_pts, replace=True)
            if self.input_ == "point cloud":
                Xboot = X[idxs,:]
            if self.input_ == "distance matrix":
                Xboot = X[idxs,:][:,idxs]
            filters_boot = self.filters_[idxs,:] if self.filters_.shape[0] > 1 else self.filters_
            colors_boot  = self.colors_[idxs,:]  if self.colors_.shape[0]  > 1 else self.colors_
            resolutions_boot, gains_boot, clustering_boot = self.resolutions_, self.gains_, self.clustering_ 
            Mboot = self.__class__(filters=filters_boot, colors=colors_boot, resolutions=resolutions_boot, gains=gains_boot, clustering=clustering_boot).fit(Xboot)
            dgm1, dgm2 = self.persistence_diagram(), Mboot.persistence_diagram()
            ndg, df = len(dgm1), 0
            for nd in range(ndg):
                npts1, npts2 = len(dgm1[nd]), len(dgm2[nd])
                D1, D2 = [], []
                for pt in range(npts1):
                    if dgm1[nd][pt][0] <= 1:
                        D1.append([dgm1[nd][pt][1][0], dgm1[nd][pt][1][1]])
                for pt in range(npts2):
                    if dgm2[nd][pt][0] <= 1:
                        D2.append([dgm2[nd][pt][1][0], dgm2[nd][pt][1][1]])
                D1, D2 = np.array(D1), np.array(D2)
                bottle = WassersteinDistance(wasserstein=np.inf).fit([D1])
                df = max(df, bottle.transform([D2])[0][0])
            distribution.append(df)
        return distribution
            
        

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

    def fit(self, X, y=None):

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

    def print_result(self, output_type="txt"):
        if output_type == "txt":
            self.cc.write_info()
        if output_type == "dot":
            self.cc.plot_dot()
        if output_type == "off":
            self.cc.plot_off()

    def compute_p_value(self, bootstrap=10):
        self.cc.compute_distribution(bootstrap)
        return self.cc.compute_p_value()

    def compute_confidence_level_from_distance(self, bootstrap=10, distance=1.0):
        self.cc.compute_distribution(bootstrap)
        return self.cc.compute_confidence_level_from_distance(distance)

    def compute_distance_from_confidence_level(self, bootstrap=10, alpha=0.1):
        self.cc.compute_distribution(bootstrap)
        return self.cc.compute_distance_from_confidence_level(alpha)

    def subpopulation(self, node_index=0):
        return self.cc.subpopulation(node_index)

    def persistence_diagram(self):
        return self.cc.compute_PD()
