import numpy as np
import itertools
import sys, os
from scipy.linalg import inv
#np.random.seed(1000)

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
    return np.sum(error ** 2 * sample_weight)

def mse_overparameter(train_data, test_majority, test_minority, nodes = 100, weighted = True):
    
    # Build model
    x, y, sample_weight = train_data
    _, input_shape = x.shape
    w = np.load(f'weights/w_{input_shape}_{nodes}.npy') #np.random.normal(size = (input_shape, nodes))
    z = hidden_layer(x, w)


    if weighted:
        z_w = z * np.sqrt(sample_weight.reshape((-1, 1)))
        y_w = y * np.sqrt(sample_weight.reshape((-1, 1)))
        beta = np.linalg.lstsq(z_w, y_w)[0]
    
    else:
        beta = np.linalg.lstsq(z, y)[0]
    

    # Evaluate 
    return evaluate((x, y), weights = [w, beta]),\
         evaluate((x, y), weights = [w, beta], sample_weight = sample_weight),\
         evaluate(test_majority, weights = [w, beta]),\
         evaluate(test_minority, weights = [w, beta])


n, p = 1000, 0.9
n1, n2 = int(n * p), int(n * (1-p))
SNRs = [0.01, 0.1, 1, 10, 100]
sigma = 0.1


iteration = int(float(sys.argv[1]))
gammas = np.logspace(0, 4 , num = 10)
nodes_list = np.rint(10 * gammas).astype('int')


if not os.path.exists('temp/'):
    os.mkdir('temp/')


for SNR in SNRs:
    # x1, x2 = np.random.normal(size = (n1, 10)), np.random.normal(size = (n2, 10))
    # beta, delta = np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.sqrt(1)*np.array([0] * 5 + [1]*5).reshape((-1,1))
    # beta, delta =  5 * beta / np.linalg.norm(beta), (1/SNR) * delta / np.linalg.norm(delta)
    # y1, y2 = x1 @ (beta + delta) + sigma * np.random.normal(size=(n1, 1)), x2 @ (beta - delta) + sigma * np.random.normal(size = (n2, 1))
    # sample_weights = np.array([(1-p)] * n1 + [p] * n2)
    # train_data = np.vstack((x1, x2)), np.vstack((y1, y2)), sample_weights


    # x_test = np.random.normal(size = (1000, 10))
    # y_test = x_test @ (beta - delta) + sigma * np.random.normal(size = (1000, 1))
    # test_minority = x_test, y_test

    # x_test = np.random.normal(size = (1000, 10))
    # y_test = x_test @ (beta + delta) + sigma * np.random.normal(size = (1000, 1))
    # test_majority = x_test, y_test




    # with open(f'temp/mse_{iteration}_{SNR}_same.txt', 'w') as f:
    #     for nodes in nodes_list:
    #         train_mse, train_mse_bal, majority_mse, minority_mse = mse_overparameter(train_data, test_majority,\
    #             test_minority, nodes=int(nodes), weighted=False)
    #         output = {'optimization': 'ERM', 'nodes': nodes, 'SNR': SNR, 'train-mse':train_mse,\
    #             'train-mse-bal': train_mse_bal, 'majority-mse': majority_mse,\
    #                 'minority-mse': minority_mse, 'trainable': 'last-layer', 'setup': 'same-core'}
    #         f.writelines(str(output)+"\n")


    #         train_mse, train_mse_bal, majority_mse, minority_mse = mse_overparameter(train_data, test_majority,\
    #             test_minority, nodes=int(nodes), weighted=True)
    #         output = {'optimization': 'weighted-ERM', 'nodes': nodes, 'SNR': SNR, 'train-mse':train_mse,\
    #             'train-mse-bal': train_mse_bal, 'majority-mse': majority_mse,\
    #                 'minority-mse': minority_mse, 'trainable': 'last-layer', 'setup': 'same-core'}
    #         f.writelines(str(output)+"\n")




    x1, x2 = np.random.normal(size = (n1, 10)), np.random.normal(size = (n2, 10))
    beta, delta = np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.sqrt(1)*np.array([0] * 5 + [1]*5).reshape((-1,1))
    beta, delta =  5 * beta / np.linalg.norm(beta), (1/SNR) * delta / np.linalg.norm(delta)
    y1, y2 = x1 @ (beta + delta) + sigma * np.random.normal(size=(n1, 1)), x2 @ (beta) + sigma * np.random.normal(size = (n2, 1))
    sample_weights = np.array([(1-p)] * n1 + [p] * n2)
    train_data = np.vstack((x1, x2)), np.vstack((y1, y2)), sample_weights


    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta) + sigma * np.random.normal(size = (1000, 1))
    test_minority = x_test, y_test

    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta + delta) + sigma * np.random.normal(size = (1000, 1))
    test_majority = x_test, y_test




    with open(f'temp/mse_{iteration}_{SNR}_diff.txt', 'w') as f:
        for nodes in nodes_list:
            train_mse, train_mse_bal, majority_mse, minority_mse = mse_overparameter(train_data, test_majority,\
                test_minority, nodes=int(nodes), weighted=False)
            output = {'optimization': 'ERM', 'nodes': nodes, 'SNR': SNR, 'train-mse':train_mse,\
                'train-mse-bal': train_mse_bal, 'majority-mse': majority_mse,\
                    'minority-mse': minority_mse, 'trainable': 'last-layer', 'setup': 'diff-core'}
            f.writelines(str(output)+"\n")
    
            train_mse, train_mse_bal, majority_mse, minority_mse = mse_overparameter(train_data, test_majority,\
                test_minority, nodes=int(nodes), weighted=True)
            output = {'optimization': 'weighted-ERM', 'nodes': nodes, 'SNR': SNR, 'train-mse':train_mse,\
                'train-mse-bal': train_mse_bal, 'majority-mse': majority_mse,\
                    'minority-mse': minority_mse, 'trainable': 'last-layer', 'setup': 'diff-core'}
            f.writelines(str(output)+"\n")
    

