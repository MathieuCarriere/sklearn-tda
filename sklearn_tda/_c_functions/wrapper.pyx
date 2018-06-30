from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef public double call_ker(obj, pair[double,double] a, pair[double,double] b):
    return obj(a,b)

cdef public double call_wei(obj, pair[double,double] a):
    return obj(a)

