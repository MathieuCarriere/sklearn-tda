# sklearn-tda: a scikit-learn compatible python package for Machine Learning and TDA

**Author**: Mathieu Carrière.

**Warning**: this code is no longer maintained since it is now part of the Gudhi library (except for Mapper and Tomato) as the representations python module: see https://github.com/GUDHI/gudhi-devel and https://github.com/GUDHI/gudhi-devel/tree/master/src/python/gudhi/representations. I recommend anyone willing to use this code to check Gudhi instead.

# Description

`sklearn_tda` is a python package for handling collections of persistence diagrams for machine learning purposes.
Various preprocessing methods, vectorizations methods and kernels for persistence diagrams are implemented in a [`scikit-learn`](http://scikit-learn.org/) compatible fashion.
Clustering methods from TDA (Mapper and ToMATo) are also implemented.

### Preprocessing

Currently available classes are: 

  * **BirthPersistenceTransform**: apply the affine transformation (x,y) -> (x,y-x) to the diagrams.

    Parameters: None.

  * **DiagramScaler**: apply scaler(s) to the diagrams (such as scalers from [scikit-learn](http://scikit-learn.org/)).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    | **use** = False | Whether to use the class or not. |
    | **scalers** = [] | List of scalers to be fit on the diagrams. Each element is a tuple whose first element is a list of coordinates and second element is a scaler (such as sklearn.preprocessing.MinMaxScaler()) for these coordinates.  |

  * **ProminentPoints**: remove points close to the diagonal.

    Parameters:

    | **name** | **description** |
    | --- | --- |
    | **use** = False|     Whether to use the class or not. |
    | **num_pts** = 10|    Cardinality threshold. |
    | **threshold** = -1|  Distance-to-diagonal threshold. |
    | **location** = "upper"|  Whether to keep the points above ("upper") or below ("lower") the previous thresholds. |

  * **Padding**: add dummy points to each diagram so that they all have the same cardinality. All points are given an additional coordinate
    indicating if the point was added after padding (0) or already present before applying this class (1).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    | **use** = False|     Whether to use the class or not. |
    

  * **DiagramSelector**: return the finite or essential points of the diagrams.

     Parameters:

    | **name** | **description** |
    | --- | --- |
    | **use** = False|     Whether to use the class or not. |
    | **limit** = np.inf | Diagram points with ordinate equal to **limit** will be considered as essential. |
    | **point_type** = "finite"| Specifies the point type to return. Either "finite" or "essential". |

### Vectorizations


Currently available classes are:

  * **Landscape**: implementation of [landscapes](http://jmlr.org/papers/v16/bubenik15a.html).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**num_landscapes** = 5| Number of landscapes.|
    |**resolution** = 100| Number of sample points of each landscape.|
    |**sample_range** = [np.nan, np.nan]| Range of each landscape. If np.nan, it is set to min and max of the diagram coordinates.|

  * **PersistenceImage**: implementation of [persistence images](http://jmlr.org/papers/v18/16-337.html).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**bandwidth** = 1.0 | Bandwidth of Gaussian kernel on the plane.|
    |**weight** = lambda x: 1| Weight on diagram points. It is a python function.|
    |**resolution** = [20,20]| Resolution of image.|
    |**im_range** = [np.nan, np.nan, np.nan, np.nan]| Range of coordinates. If np.nan, it is set to min and max of the diagram coordinates.|

  * **BettiCurve**: implementation of [Betti curves](https://www.researchgate.net/publication/316604237_Time_Series_Classification_via_Topological_Data_Analysis).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**resolution** = 100| Number of sample points of Betti curve.|
    |**sample_range** = [np.nan, np.nan]| Range of Betti curve. If np.nan, it is set to min and max of the diagram coordinates.|

  * **Silhouette**: implementation of [silhouettes](http://jocg.org/index.php/jocg/article/view/203).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**weight** = lambda x: 1| Weight on diagram points. It is a python function.|
    |**resolution** = 100| Number of sample points of silhouette.|
    |**sample_range** = [np.nan, np.nan]| Range of silhouette. If np.nan, it is set to min and max of the diagram coordinates.|

  * **TopologicalVector**: implementation of [distance vectors](https://diglib.eg.org/handle/10.1111/cgf12692).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**threshold** = 10| Number of distances to keep.|

  * **ComplexPolynomial**: implementation of [complex polynomials](https://link.springer.com/chapter/10.1007%2F978-3-319-23231-7_27).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**F** = "R"| Complex transformation to apply on the diagram points. Either "R", "S" or "T". |
    |**threshold** = 10| Number of coefficients to keep. |

  * **Entropy**: implementation of [persistence entropy](https://arxiv.org/pdf/1803.08304.pdf).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**mode** = "scalar"| Whether to compute the entropy statistic or the entropy summary function. Either "scalar" or "vector". |
    |**normalized** = True| Whether to normalize the entropy summary function.|
    |**resolution** = 100| Number of sample points of entropy summary function.|
    |**sample_range** = [np.nan, np.nan]| Range of entropy summary function. If np.nan, it is set to min and max of the diagram coordinates.|

### Kernels

Currently available classes are:

  * **PersistenceScaleSpaceKernel**: implementation of [Persistence Scale Space Kernel](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Reininghaus_A_Stable_Multi-Scale_2015_CVPR_paper.pdf).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**bandwidth** = 1.0| Bandwidth of kernel.|
    |**kernel_approx** = None| Kernel approximation method, such as [those in scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation).| 
  * **PersistenceWeightedGaussianKernel**: implementation of [Persistence Weighted Gaussian Kernel](http://proceedings.mlr.press/v48/kusano16.html).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**bandwidth** = 1.0 | Bandwidth of Gaussian kernel.|
    |**weight** = lambda x: 1| Weight on diagram points. It is a python function.|
    |**kernel_approx** = None| Kernel approximation method, such as [those in scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation).|
    |**use_pss** = False| Whether to add symmetric of points from the diagonal.|

  * **SlicedWassersteinKernel**: implementation of [Sliced Wasserstein Kernel](http://proceedings.mlr.press/v70/carriere17a.html).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**num_directions** = 10| Number of directions.|
    |**bandwidth** = 1.0| Bandwidth of kernel.|

  * **PersistenceFisherKernel**: implementation of [Persistence Fisher Kernel](papers.nips.cc/paper/8205-persistence-fisher-kernel-a-riemannian-manifold-kernel-for-persistence-diagrams).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**bandwidth_fisher** = 1.0| Bandwidth of Gaussian kernel for Fisher distance.|
    |**bandwidth** = 1.0| Bandwidth of kernel.|
    |**kernel_approx** = None| Kernel approximation method, such as [those in scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation).|

### Metrics

Currently available classes are:

  * **BottleneckDistance**: wrapper for bottleneck distance module of Gudhi. **Requires Gudhi!!**

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**epsilon** = 0.001| Approximation error.|

  * **SlicedWassersteinDistance**: implementation of [Sliced Wasserstein distance](http://proceedings.mlr.press/v70/carriere17a.html).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**num_directions** = 10| Number of directions.|

  * **PersistenceFisherDistance**: implementation of [Fisher Information distance](papers.nips.cc/paper/8205-persistence-fisher-kernel-a-riemannian-manifold-kernel-for-persistence-diagrams).

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**bandwidth** = 1.0| Bandwidth of Gaussian kernel.|
    |**kernel_approx** = None| Kernel approximation method, such as [those in scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation).|

### Clustering

Currently available classes are: 

  * **MapperComplex**: implementation of the [Mapper](https://research.math.osu.edu/tgda/mapperPBG.pdf). **Requires Gudhi!!**. Note that further statistical analysis can be performed with [statmapper](https://github.com/MathieuCarriere/statmapper).

    Parameters

    | **name** | **description** |
    | --- | --- |
    | **filters**| Numpy array specifying the filter values. Each row is a point and each column is a filter dimension.|
    | **filter_bnds**| Numpy array specifying the lower and upper limits of each filter. If NaN, they are automatically computed.  |
    | **colors**| Numpy array specifying the color values. Each row is a point and each column is a color dimension. |
    | **resolutions**| List of resolutions for each filter dimension. If NaN, they are computed automatically. |
    | **gains**| List of gains for each filter dimension. |
    | **clustering** = sklearn.cluster.DBSCAN()| Clustering method. |
    | **input** = "point cloud" | String specifying input type. Either "point cloud" or "distance matrix". |
    | **mask** = 0| Threshold on the node sizes.|

  * **ToMATo**: implementation of [ToMATo](https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/). **Requires Gudhi!!**

    Parameters

    | **name** | **description** |
    | --- | --- |
    | **tau** = None| Merging parameter. If None, n_clusters is used. |
    | **n_clusters** = None| Number of clusters. If None, it is automatically computed. |
    | **density_estimator** = DistanceToMeasure()| Density estimator method |
    | **n_neighbors** = None| Number of neighbors for k-neighbors graph. If None, radius is used.|
    | **radius** = None| Threshold for delta-neighborhood graph. If None, it is automatically computed. |
    | **verbose** = False| Print info.|

  * **DistanceToMeasure**: implementation of [distance-to-measure](https://arxiv.org/pdf/1412.7197.pdf) density estimator.

    Parameters

    | **name** | **description** |
    | --- | --- |
    | **n_neighbors** = 30| Number of nearest neighbors.|

# Installing sklearn_tda

The sklearn_tda library requires:

* python [>=2.7, >=3.5]
* numpy [>= 1.8.2]
* scikit-learn

For now, the package has to be compiled from source. You have to 

* download the code with:
```shell
git clone https://github.com/MathieuCarriere/sklearn_tda
```
* move to the directory:
```shell
cd sklearn_tda
```
* compile with:
```shell
(sudo) pip install .
```

The package can then be imported in a python shell with:
```shell
import sklearn_tda
``` 

Usage
=====

All modules are standard scikit-learn modules: they have fit, transform and fit_transform methods.
Hence, the most common way to use module X is to call X.fit_transform(input).
The input of all modules (except the clustering modules) are lists of persistence diagram, which are represented as lists of 2D numpy arrays.
Various examples can be found [here](example/).

