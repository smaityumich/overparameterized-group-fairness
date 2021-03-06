import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import itertools
import sys, os

def mse_overparameter(train_data, test_majority, test_minority, nodes = 100, optimizer = 'sgd', l2_reg = 0.00001,\
     epochs = 40):
    
    # Build model
    x, y, sample_weight = train_data
    _, input_shape = x.shape
    l2_reg = 0.001/epochs
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    hidden = tf.keras.layers.Dense(nodes, activation="relu", name="hidden", \
 kernel_regularizer=tf.keras.regularizers.L2(l2_reg),\
        bias_regularizer=tf.keras.regularizers.L2(l2_reg), trainable = False)
    outputs = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2_reg),\
        bias_regularizer=tf.keras.regularizers.L2(l2_reg))(hidden(inputs))
    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)

    # Compile and fit
    model.compile(optimizer=optimizer, loss = 'mse')
    model.fit(x, y, epochs = epochs, verbose = 0, sample_weight = sample_weight)

    ## Fit and then compile
    # Use over-parametrized solution

    # Evaluate 
    return model.evaluate(x, y, verbose = 1),\
         model.evaluate(x, y, sample_weight = sample_weight, verbose = 1),\
         model.evaluate(*test_majority, verbose = 1),\
         model.evaluate(*test_minority, verbose = 1)


n, p = 2000, 0.9
n1, n2 = int(n * p), int(n * (1-p))
beta_norms = [0.1]


iteration = int(float(sys.argv[1]))
nodes_list = 10* 4 **np.array(range(6))
epochs_list = [50, 5000]

if not os.path.exists('temp/'):
    os.mkdir('temp/')


for beta_norm in beta_norms:
    x1, x2 = np.random.normal(size = (n1, 10)), np.random.normal(size = (n2, 10))
    #beta, delta = 10* np.array([1/np.sqrt(10)] * 10).reshape((-1,1)), 10*np.array([-2/np.sqrt(10)] + [0]*9).reshape((-1,1))
    beta, delta = beta_norm * np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.sqrt(1)*np.array([0] * 5 + [1]*5).reshape((-1,1))
    y1, y2 = x1 @ (beta +1* delta) + 0.1 * np.random.normal(size=(n1, 1)), x2 @ (beta - delta) + 0.1 * np.random.normal(size = (n2, 1))
    sample_weights = np.array([(1-p)] * n1 + [p] * n2)
    train_data = np.vstack((x1, x2)), np.vstack((y1, y2)), sample_weights


    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta - delta) + 0.1 * np.random.normal(size = (1000, 1))
    test_minority = x_test, y_test

    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta + 1*delta) + 0.1 * np.random.normal(size = (1000, 1))
    test_majority = x_test, y_test




    with open(f'temp/mse_{iteration}_{beta_norm}_same.txt', 'w') as f:
        for nodes, epochs in itertools.product(nodes_list, epochs_list):
            train_mse, train_mse_bal, majority_mse, minority_mse = mse_overparameter(train_data, test_majority,\
                test_minority, nodes=int(nodes), optimizer='Adam', epochs=epochs)
            output = {'nodes': nodes, 'beta-norm': beta_norm, 'epochs': epochs, 'train-mse':train_mse,\
                'train-mse-bal': train_mse_bal, 'majority-mse': majority_mse,\
                    'minority-mse': minority_mse, 'trainable': 'last-layer', 'setup': 'same-core'}
            f.writelines(str(output)+"\n")

    x1, x2 = np.random.normal(size = (n1, 10)), np.random.normal(size = (n2, 10))
    #beta, delta = 10* np.array([1/np.sqrt(10)] * 10).reshape((-1,1)), 10*np.array([-2/np.sqrt(10)] + [0]*9).reshape((-1,1))
    beta, delta = beta_norm * np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.sqrt(1)*np.array([0] * 5 + [1]*5).reshape((-1,1))
    y1, y2 = x1 @ (beta) + 0.1 * np.random.normal(size=(n1, 1)), x2 @ (beta + delta) + 0.1 * np.random.normal(size = (n2, 1))
    sample_weights = np.array([(1-p)] * n1 + [p] * n2)
    train_data = np.vstack((x1, x2)), np.vstack((y1, y2)), sample_weights


    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta + delta) + 0.1 * np.random.normal(size = (1000, 1))
    test_minority = x_test, y_test

    x_test = np.random.normal(size = (1000, 10))
    y_test = x_test @ (beta) + 0.1 * np.random.normal(size = (1000, 1))
    test_majority = x_test, y_test




    with open(f'temp/mse_{iteration}_{beta_norm}_s.txt', 'w') as f:
        for nodes, epochs in itertools.product(nodes_list, epochs_list):
            train_mse, train_mse_bal, majority_mse, minority_mse = mse_overparameter(train_data, test_majority,\
                test_minority, nodes=int(nodes), optimizer='Adam', epochs=epochs)
            output = {'nodes': nodes, 'beta-norm': beta_norm, 'epochs': epochs, 'train-mse':train_mse,\
                'train-mse-bal': train_mse_bal, 'majority-mse': majority_mse,\
                    'minority-mse': minority_mse, 'trainable': 'last-layer', 'setup': 'diff-core'}
            f.writelines(str(output)+"\n")
    

