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
    double                 sw            (vector[pair[double, double]],          vector[pair[double, double]],          double, int)
    vector[vector[double]] sw_matrix     (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  double, int)
    double                 pwg           (vector[pair[double, double]],          vector[pair[double, double]],          const KernelWrapper&,    const WeightWrapper&)
    vector[vector[double]] pwg_matrix    (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  const KernelWrapper&,    const WeightWrapper&)

def sliced_wasserstein(diagram_1, diagram_2, sigma = 1, N = 100):
    return sw(diagram_1, diagram_2, sigma, N)

def sliced_wasserstein_matrix(diagrams_1, diagrams_2, sigma = 1, N = 100):
    return sw_matrix(diagrams_1, diagrams_2, sigma, N)

def persistence_weighted_gaussian(diagram_1, diagram_2, kernel = lambda x, y: (x[0]*y[0]) + (x[1]-y[1]), weight = lambda x: 1):
    cdef KernelWrapper k = KernelWrapper(kernel)
    cdef WeightWrapper w = WeightWrapper(weight)
    return pwg(diagram_1, diagram_2, k, w)

def persistence_weighted_gaussian_matrix(diagrams_1, diagrams_2, kernel = lambda x, y: (x[0]*y[0]) + (x[1]-y[1]), weight = lambda x: 1):
    cdef KernelWrapper k = KernelWrapper(kernel)
    cdef WeightWrapper w = WeightWrapper(weight)
    return pwg_matrix(diagrams_1, diagrams_2, k, w)
