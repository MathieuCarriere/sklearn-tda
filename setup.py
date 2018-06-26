from __future__ import print_function
import sys
from warnings import warn
from platform import system
from setuptools import setup, find_packages, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_address = "./sklearn_tda/_c_functions/"
vect = Extension(name="sklearn_tda._c_vectors",
                sources=[ext_address + "vectors.pyx"],
                language="c++",
                extra_compile_args=["-std=c++14"], 
                extra_link_args=["-std=c++11"])

kern = Extension(name="sklearn_tda._c_kernels",
                sources=[ext_address + "kernels.pyx"],
                language="c++",
                extra_compile_args=["-std=c++14"], 
                extra_link_args=["-std=c++11"])

try:
    from Cython.Distutils import build_ext
    modules = [vect, kern]
except ImportError:
    modules = []

setuptools.setup(
    name="sklearn_tda",
    version="2",
    author="Mathieu Carriere",
    author_email="mathieu.carriere3@gmail.com",
    description="A scikit-learn compatible package for doing Machine Learning and TDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    ext_modules = modules,
)
