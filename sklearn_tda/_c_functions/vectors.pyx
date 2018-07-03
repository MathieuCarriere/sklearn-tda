from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string

cdef extern from "wrapper_class.h":
    cdef cppclass WeightWrapper:
        WeightWrapper()
        WeightWrapper(object)

cdef extern from "vectors/Vectors_interface.h":
    vector[vector[double]] compute_ls    (vector[pair[double, double]], int, double, double, int)
    vector[vector[double]] compute_pim   (vector[pair[double, double]], double, double, int, double, double, int, double, WeightWrapper)
    vector[double]         compute_sh    (vector[pair[double, double]], double, double, int, WeightWrapper)
    vector[int]            compute_bc    (vector[pair[double, double]], double, double, int)

def landscape(diagram, nb_ls, min_x, max_x, res_x):
    return compute_ls(diagram, nb_ls, min_x, max_x, res_x)

def persistence_image(diagram, min_x, max_x, res_x, min_y, max_y, res_y, sigma, weight):
    cdef WeightWrapper w = WeightWrapper(weight)
    return compute_pim(diagram, min_x, max_x, res_x, min_y, max_y, res_y, sigma, w)

def silhouette(diagram, min_x, max_x, res_x, weight):
    cdef WeightWrapper w = WeightWrapper(weight)
    return compute_sh(diagram, min_x, max_x, res_x, w)

def betti_curve(diagram, min_x, max_x, res_x):
    return compute_bc(diagram, min_x, max_x, res_x)
