from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "interface_bottleneck.h":
    double bottleneck_dist(vector[pair[double, double]], vector[pair[double, double]], double)

def bottleneck(diagram_1, diagram_2, delta = 0.01):
    return bottleneck_dist(diagram_1, diagram_2, delta)
