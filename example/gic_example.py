import numpy as np
import sklearn_tda as tda

X = np.loadtxt("human")

cov = tda.GraphInducedComplex(cover_type = "functional", filter = 2).fit(X)
cov.print_result("dot")

print(cov.compute_confidence_level_from_distance(bootstrap = 20, distance = 0.2))
print(cov.compute_distance_from_confidence_level(bootstrap = 20, alpha = 0.95))
print(cov.compute_p_value(bootstrap = 20))
