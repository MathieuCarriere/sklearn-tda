import numpy as np
from sklearn.metrics import pairwise_distances
import os
import gudhi as gd
from sklearn_tda import *

X = np.loadtxt("inputs/human")


print("Mapper computation with point cloud")
mapper = MapperComplex(inp="point cloud", filters=X[:,[2,0]], filter_bnds=np.array([[np.nan,np.nan],[np.nan,np.nan]]), resolutions=np.array([np.nan,np.nan]), gains=np.array([0.33,0.33]), colors=X[:,2:3]).fit(X)
print(mapper.mapper_.get_filtration())


print("Mapper computation with pairwise distances only")
X = pairwise_distances(X)
mapper = MapperComplex(inp="distance matrix", filters=X[:,[2,0]], filter_bnds=np.array([[np.nan,np.nan],[np.nan,np.nan]]), resolutions=np.array([np.nan,np.nan]), gains=np.array([0.33,0.33]), colors=np.max(X, axis=1)[:,np.newaxis]).fit(X)
print(mapper.mapper_.get_filtration())
