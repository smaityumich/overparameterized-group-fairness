import numpy as np
np.random.seed(1)
d = 10



def sample_from_sphere(n, d):
    x = np.random.normal(size = (n, d))
    x = (x.T/np.linalg.norm(x, axis = 1)).T
    return x


n = 1000
gammas = np.logspace(0.01, 1.2 , num = 10)
nodes_upper = np.rint(n * gammas).astype('int')

gammas = np.logspace(0.01, 2 , num = 10)
nodes_lower = np.rint(n / gammas).astype('int')

nodes_list = np.append(nodes_lower, nodes_upper)
for nodes in nodes_list:
    w = sample_from_sphere(d, nodes)
    np.save(f'weights/w_{d}_{nodes}.npy', w)

