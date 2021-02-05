import numpy as np
import itertools
import sys, os
from scipy.linalg import inv
#np.random.seed(1000)

def hidden_layer(x, w):
    z = x @ w
    z[z<0]=0
    #z = z - np.mean(z, axis=0)
    return z

def predict(x, weights):
    w, beta = weights
    z = hidden_layer(x, w)
    return z @ beta

def evaluate(data, weights, sample_weight = []):
    x, y = data
    n, _ = x.shape
    if sample_weight == []:
        sample_weight = np.array([1]*n)
    sample_weight = sample_weight/np.sum(sample_weight)
    y_hat = predict(x, weights)
    error = (y - y_hat).reshape((-1,))
    return np.sum(error ** 2 * sample_weight)

def mse(data, nodes = 100):
    train_data, test_data = data

    x, y = train_data
    _, input_shape = x.shape
    w = np.load(f'weights/w_{input_shape}_{nodes}.npy') 
    z = hidden_layer(x, w)
    beta = np.linalg.lstsq(z, y)[0]

    return evaluate(test_data, weights=[w, beta])


def approx_error(data_misspecified, data_accurate, nodes = 100):
    return mse(data_misspecified, nodes=nodes) - mse(data_accurate, nodes = nodes)


n, p = 1000, 0.9
n1, n2 = int(n * p), int(n * (1-p))
SNRs = [0.01, 0.1, 1, 10, 100]
sigma = 0.1


iteration = int(float(sys.argv[1]))


gammas = np.logspace(0.01, 1.2 , num = 10)
nodes_upper = np.rint(n * gammas).astype('int')

gammas = np.logspace(0.01, 2 , num = 10)
nodes_lower = np.rint(n / gammas).astype('int')

nodes_list = np.append(nodes_lower, nodes_upper)


if not os.path.exists('ERM-beta/'):
    os.mkdir('ERM-beta/')


for SNR in SNRs:
    
    x1, x2 = np.random.normal(size = (n1, 10)), np.random.normal(size = (n2, 10))
    beta, delta = np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.sqrt(1)*np.array([0] * 5 + [1]*5).reshape((-1,1))
    beta, delta =   SNR * beta / np.linalg.norm(beta),  delta / np.linalg.norm(delta)
    y1, y2 = x1 @ (beta + delta) + sigma * np.random.normal(size=(n1, 1)), x2 @ (beta) + sigma * np.random.normal(size = (n2, 1))
    train_misspecified = np.vstack((x1, x2)), np.vstack((y1, y2))


    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta) + sigma * np.random.normal(size = (1000, 1))
    test_minority = x_test, y_test

    x = np.random.normal(size = (n, 10))
    y = x @ beta + sigma * np.random.normal(size = (n, 1))
    train_accurate = x_test, y_test

    data_misspecified = train_misspecified, test_minority
    data_accurate = train_accurate, test_minority




    with open(f'ERM-beta/beta_{iteration}_{SNR}.txt', 'w') as f:
        for nodes in nodes_list:
            mse_misspecified = mse(data_misspecified, nodes=nodes)
            mse_accurate = mse(data_accurate, nodes = nodes)
            approx_error = mse_misspecified - mse_accurate
            
            output = {'optimization': 'ERM', 'nodes': nodes, 'SNR': SNR, 'missp-mse':mse_misspecified,\
                'acc-mse': mse_accurate, 'approx-error': approx_error}
            f.writelines(str(output)+"\n")
            
    
    

