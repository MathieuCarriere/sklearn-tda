from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string

cdef extern from "wrapper_class.h":
    cdef cppclass KernelWrapper:
        KernelWrapper()
        KernelWrapper(object)

cdef extern from "wrapper_class.h":
    cdef cppclass WeightWrapper:
        WeightWrapper()
        WeightWrapper(object)

cdef extern from "kernels/Kernels_interface.h":
    vector[vector[double]] swk_matrix     (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  double, int)
    vector[vector[double]] sw_matrix      (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  int)
    vector[vector[double]] pwgk_matrix    (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  double, WeightWrapper)

def sliced_wasserstein_kernel_matrix(diagrams_1, diagrams_2, sigma, N):
    return swk_matrix(diagrams_1, diagrams_2, sigma, N)

def sliced_wasserstein_matrix(diagrams_1, diagrams_2, N):
    return sw_matrix(diagrams_1, diagrams_2, N)

def persistence_weighted_gaussian_kernel_matrix(diagrams_1, diagrams_2, sigma, weight):
    cdef WeightWrapper w = WeightWrapper(weight)
    return pwgk_matrix(diagrams_1, diagrams_2, sigma, w)
