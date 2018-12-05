from setuptools import setup, find_packages, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

hera_w = Extension(name                = "sklearn_tda.hera_wasserstein",
                   sources             = ["./sklearn_tda/hera_wasserstein.pyx"],
                   language            = "c++",
                   extra_compile_args  = ["-std=c++14", "-I./sklearn_tda/hera/geom_matching/wasserstein/include/"])

hera_b = Extension(name                = "sklearn_tda.hera_bottleneck",
                   sources             = ["./sklearn_tda/hera_bottleneck.pyx"],
                   language            = "c++",
                   extra_compile_args  = ["-std=c++14", "-I./sklearn_tda/hera/geom_bottleneck/include/"])

try:
    from Cython.Distutils import build_ext
    from Cython.Build     import cythonize
    modules, cmds = cythonize([hera_w, hera_b]), {"build_ext": build_ext}
    print("Cython found")

except ImportError:
    modules, cmds = [], {}
    print("Cython not found")

setup(
    name                           = "sklearn_tda",
    version                        = "0",
    author                         = "Mathieu Carriere",
    author_email                   = "mathieu.carriere3@gmail.com",
    description                    = "A scikit-learn compatible package for doing Machine Learning and TDA",
    long_description               = long_description,
    long_description_content_type  = "text/markdown",
    url                            = "https://github.com/MathieuCarriere/sklearn_tda/",
    packages                       = find_packages(),
    classifiers                    = ("Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"),
    ext_modules                    = modules,
    cmdclass                       = cmds,
)
