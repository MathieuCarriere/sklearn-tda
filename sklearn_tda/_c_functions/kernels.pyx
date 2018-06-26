from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
import os

cdef extern from "Kernels_interface.h":
    double                 sw            (vector[pair[double, double]],          vector[pair[double, double]],          double, int)
    vector[vector[double]] sw_matrix     (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  double, int)
    double                 pwg           (vector[pair[double, double]],          vector[pair[double, double]],          string, string, double, double, double)
    vector[vector[double]] pwg_matrix    (vector[vector[pair[double, double]]],  vector[vector[pair[double, double]]],  string, string, double, double, double)

def sliced_wasserstein(diagram_1, diagram_2, sigma = 1, N = 100):
    return sw(diagram_1, diagram_2, sigma, N)

def sliced_wasserstein_matrix(diagrams_1, diagrams_2, sigma = 1, N = 100):
    return sw_matrix(diagrams_1, diagrams_2, sigma, N)

def persistence_weighted_gaussian(diagram_1, diagram_2, kernel = "rbf", weight = "linear", sigma = 1.0, c = 1.0, d = 1.0):
    return pwg(diagram_1, diagram_2, kernel, weight, sigma, c, d)

def persistence_weighted_gaussian_matrix(diagrams_1, diagrams_2, kernel = "rbf", weight = "linear", sigma = 1.0, c = 1.0, d = 1.0):
    return pwg_matrix(diagrams_1, diagrams_2, kernel, weight, sigma, c, d)
