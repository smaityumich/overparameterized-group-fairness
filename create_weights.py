import numpy as np
np.random.seed(1)
d = 10
gammas = np.logspace(0, 3, num = 10)
nodes_list = np.rint(50 * gammas).astype('int')


def sample_from_sphere(n, d):
    x = np.random.normal(size = (n, d))
    x = (x.T/np.linalg.norm(x, axis = 1)).T
    return x


gammas = np.logspace(0, 3, num = 10)
nodes_list = np.rint(50 * gammas).astype('int')

for nodes in nodes_list:
    w = sample_from_sphere(d, nodes)
    np.save(f'weights/w_{d}_{nodes}.npy', w)

gammas = np.logspace(0, 4 , num = 10)
nodes_list = np.rint(10 * gammas).astype('int')


for nodes in nodes_list:
    w = sample_from_sphere(d, nodes)
    np.save(f'weights/w_{d}_{nodes}.npy', w)

gammas = np.logspace(0, 3, num = 10)
nodes_list = np.rint(10 * gammas).astype('int')

for nodes in nodes_list:
    w = sample_from_sphere(d, nodes)
    np.save(f'weights/w_{d}_{nodes}.npy', w)

gammas = np.logspace(0, 4 , num = 10)
nodes_list = np.rint(2 * gammas).astype('int')


for nodes in nodes_list:
    w = sample_from_sphere(d, nodes)
    np.save(f'weights/w_{d}_{nodes}.npy', w)
