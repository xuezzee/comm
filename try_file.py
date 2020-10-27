import numpy as np

x = np.random.poisson(lam=3, size=20000)
print(x)
print(x.sum())