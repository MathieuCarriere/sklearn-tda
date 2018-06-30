from __future__ import print_function
import sys
from warnings import warn
from platform import system
import setuptools
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_address = "./sklearn_tda/_c_functions/"
vect = Extension(name="sklearn_tda.vectors",
                 sources=[ext_address + "vectors.pyx", ext_address + "wrapper.pyx"],
                 language="c++",
                 extra_compile_args=["-std=c++14"],
                 extra_link_args=["-std=c++11"])

kern = Extension(name="sklearn_tda.kernels",
                 sources=[ext_address + "kernels.pyx", ext_address + "wrapper.pyx"],
                 language="c++",
                 extra_compile_args=["-std=c++14"],
                 extra_link_args=["-std=c++11"])

hera_w = Extension(name="sklearn_tda.hera_wasserstein",
                 sources=[ext_address + "hera_wasserstein.pyx"],
                 language="c++",
                 extra_compile_args=["-std=c++14", "-I./sklearn_tda/_c_functions/hera/geom_matching/wasserstein/include/"],
                 extra_link_args=["-std=c++11"])

hera_b = Extension(name="sklearn_tda.hera_bottleneck",
                 sources=[ext_address + "hera_bottleneck.pyx"],
                 language="c++",
                 extra_compile_args=["-std=c++14", "-I./sklearn_tda/_c_functions/hera/geom_bottleneck/include/"],
                 extra_link_args=["-std=c++11"])

try:
    from Cython.Distutils import build_ext
    modules = [vect, kern, hera_w, hera_b]
    print("Cython found")

except ImportError:
    modules = []
    print("Cython not found")

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
    classifiers=("Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"),
    ext_modules = cythonize(modules),
    cmdclass = {"build_ext": build_ext},
)
