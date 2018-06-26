from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
import os

cdef extern from "Vectors_interface.h":
    vector[vector[double]] compute_ls    (vector[pair[double, double]], int, double, double, int)
    vector[vector[double]] compute_pim   (vector[pair[double, double]], double, double, int, double, double, int, string, string, double, double, double)

def landscape(diagram, nb_ls = 10, min_x = 0.0, max_x = 1.0, res_x = 100):
    return compute_ls(diagram, nb_ls, min_x, max_x, res_x)

def persistence_image(diagram, min_x = 0.0, max_x = 1.0, res_x = 10, min_y = 0.0, max_y = 1.0, res_y = 10, kernel = "rbf", weight = "linear", sigma = 1.0, c = 1.0, d = 1.0):
    return compute_pim(diagram, min_x, max_x, res_x, min_y, max_y, res_y, kernel, weight, sigma, c, d)
