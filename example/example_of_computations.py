import sklearn_tda as tda
import matplotlib.pyplot as plt
import numpy as np

D = np.array([[0.0,4.0],[1.0,2.0],[3.0,8.0],[6.0,8.0]])
diags = []
diags.append(D)
plt.scatter(D[:,0],D[:,1])
plt.plot([0.0,10.0],[0.0,10.0])
plt.show()

diagsT = tda.DiagramPreprocessor(tda.BirthPersistenceTransform()).fit_transform(diags)
plt.scatter(diagsT[0][:,0], diagsT[0][:,1])
plt.axis([0.0,10.0,0.0,10.0])
plt.show()

D = np.array([[1.0,5.0],[3.0,6.0],[2.0,7.0]])
diags2 = []
diags2.append(D)


SW = tda.DiagramKernelizer(name="SlicedWasserstein", N = 10, gaussian_bandwidth = 1.0)
X = SW.fit(diags)
Y = SW.transform(diags2)
print(Y)


def gauss(p,q):
  sigma = 1.0
  return np.exp( -(p[0]-q[0])*(p[0]-q[0])/(2*sigma*sigma) -(p[1]-q[1])*(p[1]-q[1])/(2*sigma*sigma) ) / (sigma*np.sqrt(2*np.pi))

def poly(p,q):
  return p[0]*q[0] + p[1]*q[1] + 1.0

PWG = tda.DiagramKernelizer(name="PersistenceWeightedGaussian", kernel = "rbf")
X = PWG.fit(diags)
Y = PWG.transform(diags2)
print(Y)

PWG = tda.DiagramKernelizer(name="PersistenceWeightedGaussian", kernel = gauss)
X = PWG.fit(diags)
Y = PWG.transform(diags2)
print(Y)

PWG = tda.DiagramKernelizer(name="PersistenceWeightedGaussian", kernel = "poly")
X = PWG.fit(diags)
Y = PWG.transform(diags2)
print(Y)

PWG = tda.DiagramKernelizer(name="PersistenceWeightedGaussian", kernel = poly)
X = PWG.fit(diags)
Y = PWG.transform(diags2)
print(Y)


LS = tda.FiniteDiagramVectorizer(name = "Landscape", resolution_x = 1000)
L = LS.fit_transform(diags)
print(L[0])
plt.plot(L[0][:1000])
plt.plot(L[0][1000:2000])
plt.plot(L[0][2000:3000])
plt.show()

BC = tda.FiniteDiagramVectorizer(name = "BettiCurve", resolution_x = 1000)
B = BC.fit_transform(diags)
plt.plot(B[0])
plt.show()

PI = tda.FiniteDiagramVectorizer(name = "PersistenceImage", kernel = "rbf", min_x = 0, max_x = 10, min_y=0, max_y=10, resolution_x = 100,  resolution_y = 100)
I = PI.fit_transform(diagsT)
print(I)
plt.imshow(np.flip(np.reshape(I[0], [100,100]), 0))
plt.show()

PI = tda.FiniteDiagramVectorizer(name = "PersistenceImage", kernel = gauss, min_x = 0, max_x = 10, min_y=0, max_y=10, resolution_x = 100, resolution_y = 100)
I = PI.fit_transform(diagsT)
print(I)
plt.imshow(np.flip(np.reshape(I[0], [100,100]), 0))
plt.show()

PI = tda.FiniteDiagramVectorizer(name = "PersistenceImage", kernel = "poly", min_x = 0, max_x = 10, min_y=0, max_y=10, resolution_x = 100, resolution_y = 100)
I = PI.fit_transform(diagsT)
print(I)
plt.imshow(np.flip(np.reshape(I[0], [100,100]), 0))
plt.show()

PI = tda.FiniteDiagramVectorizer(name = "PersistenceImage", kernel = poly, min_x = 0, max_x = 10, min_y=0, max_y=10, resolution_x = 100, resolution_y = 100)
I = PI.fit_transform(diagsT)
print(I)
plt.imshow(np.flip(np.reshape(I[0], [100,100]), 0))
plt.show()

K = np.array(  [[0.5, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 0.5]])

PI = tda.FiniteDiagramVectorizer(name = "PersistenceImage", kernel = K, min_x = 0, max_x = 10, min_y=0, max_y=10, resolution_x = 100, resolution_y = 100)
I = PI.fit_transform(diagsT)
print(I)
plt.imshow(np.flip(np.reshape(I[0], [100,100]), 0))
plt.show()
