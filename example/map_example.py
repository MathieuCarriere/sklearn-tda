import numpy as np
import sklearn_tda as tda
from sklearn.cluster import *
import graphviz as gv
import matplotlib.cm as cm
import matplotlib.colors as col

X = np.loadtxt("human")

map = tda.MapperComplex(filters = X[:,[2,0]], resolutions = [10,3], gains = [0.33,0.33], color = X[:,2], clustering = DBSCAN(eps = 0.05, min_samples = 5))
map.fit(X)

G = gv.Graph()

min_col, min_sz, max_col, max_sz = 1e10, 1e10, -1e10, -1e10
for value in map.graph_:
    if len(value[0]) == 1:
        min_col = min(min_col, value[1])
        max_col = max(max_col, value[1])
        min_sz = min(min_sz, value[2])
        max_sz = max(max_sz, value[2])

num_pts, num_edges = 0, 0
for value in map.graph_:
    if len(value[0]) == 1:
        num_pts += 1
        G.node(str(value[0][0]), style = "filled", fillcolor = col.rgb2hex(cm.rainbow(  (value[1]-min_col)/(max_col-min_col)  )[:3]), size = str((1.0*value[2]-min_sz)/(max_sz-min_sz)))
    if len(value[0]) == 2:
        num_edges += 1
        G.edge(str(value[0][0]), str(value[0][1]))

G.render("graphviz")

f = open("kepler", "w")
f.write("%s\n%s\n%s\n%f %f\n%d %d\n" % ("human", "2 coordinates", "height", 10, 0.33, num_pts, num_edges))

for value in map.graph_:
    if len(value[0]) == 1:
        f.write(str(value[0][0]) + " " + str(value[1]) + " " + str(value[2]) + "\n")

for value in map.graph_:
    if len(value[0]) == 2:
        f.write(str(value[0][0]) + " " + str(value[0][1]) + "\n")
