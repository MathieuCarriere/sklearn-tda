from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
import os

cdef extern from "hera/wass_interface.h":
    double wasserstein_dist(vector[pair[double, double]], vector[pair[double, double]], double, double)

def wasserstein(diagram_1, diagram_2, p = 1, delta = 0.01):
    return wasserstein_dist(diagram_1, diagram_2, p, delta)


