import numpy as np
import sys
import argparse
from scipy.linalg import inv
parser = argparse.ArgumentParser('Nodes')
parser.add_argument('--nodes', dest='nodes', type= int, nargs='*', default=[100])
args = parser.parse_args()


def hidden_layer(x, w):
    z = x @ w
    z[z<0]=0
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
    return np.sum(error ** 2  * sample_weight)

def mse_overparameter(train_data, test_majority, test_minority, w, nodes = 100, weighted = True):
    
    # Build model
    x, y, sample_weight = train_data
    n, _ = x.shape
    
    z = hidden_layer(x, w)
    if weighted:
        z = z * np.sqrt(sample_weight.reshape((-1, 1)))
    # Fit model
    beta = np.linalg.lstsq(z, y)[0]
    
    
    

    # Evaluate 
    return evaluate((x, y), weights = [w, beta]),\
         evaluate((x, y), weights = [w, beta], sample_weight = sample_weight),\
         evaluate(test_majority, weights = [w, beta]),\
         evaluate(test_minority, weights = [w, beta])



n, p = 200, 0.9
n1, n2 = int(n * p), int(n * (1-p))
beta_norm = 5
sigma = 0.1



x1, x2 = np.random.normal(size = (n1, 10)), np.random.normal(size = (n2, 10))
beta, delta = beta_norm * np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.sqrt(1)*np.array([0] * 5 + [1]*5).reshape((-1,1))
y1, y2 = x1 @ (beta) + sigma * np.random.normal(size=(n1, 1)), x2 @ (beta + delta) + sigma * np.random.normal(size = (n2, 1))
sample_weights = np.array([(1-p)] * n1 + [p] * n2)
train_data = np.vstack((x1, x2)), np.vstack((y1, y2)), sample_weights


x_test = np.random.normal(size = (1000, 10))
y_test = x_test @ (beta + delta) + sigma * np.random.normal(size = (1000, 1))
test_minority = x_test, y_test

x_test = np.random.normal(size = (1000, 10))
y_test = x_test @ (beta) + sigma * np.random.normal(size = (1000, 1))
test_majority = x_test, y_test

#nodes = int(float(sys.argv[1]))
for nodes in args.nodes:
    w = np.random.normal(size = (10, nodes))#/np.sqrt(nodes)
    train_mse, train_mse_bal,  majority_mse, minority_mse = mse_overparameter(train_data, test_majority, test_minority, w = w, nodes=nodes, weighted=False)
    print(f'# Hidden nodes: {nodes}\nTrain mse: {train_mse}\nBal-train mse: {train_mse_bal}\nTest mse for majority: {majority_mse}\nTest mse for minority group: {minority_mse}\n\n')

