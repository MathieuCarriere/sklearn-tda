import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

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
)
