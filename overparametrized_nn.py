import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import argparse
parser = argparse.ArgumentParser('Nodes')
parser.add_argument('--nodes', dest='nodes', type= int, nargs='*', default=[100])
args = parser.parse_args()


def mse_overparameter(train_data, test_majority, test_minority, nodes = 100, optimizer = 'sgd', l2_reg = 0.00001,\
     epochs = 400):
    
    # Build model
    x, _ = train_data
    _, input_shape = x.shape
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    hidden = tf.keras.layers.Dense(nodes, activation="relu", name="hidden", kernel_regularizer=tf.keras.regularizers.L2(l2_reg),\
        bias_regularizer=tf.keras.regularizers.L2(l2_reg), trainable = False)
    outputs = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2_reg),\
        bias_regularizer=tf.keras.regularizers.L2(l2_reg))(hidden(inputs))
    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)

    # Compile and fit
    model.compile(optimizer='sgd', loss = 'mse')
    model.fit(*train_data, epochs = epochs, verbose = 0)

    # Evaluate 
    return model.evaluate(*train_data, verbose = 0), model.evaluate(*test_majority, verbose = 0),\
         model.evaluate(*test_minority, verbose = 0)



x1, x2 = np.random.normal(size = (3000, 10)), np.random.normal(size = (300, 10))
beta, delta = 2 * np.array([1] * 5 + [0] * 5).reshape((-1,1)), np.array([0]* 5 + [1]*5).reshape((-1,1))
y1, y2 = x1 @ (beta + delta) + 0.1 * np.random.normal(size=(3000, 1)), x2 @ (beta - delta) + 0.1 * np.random.normal(size = (300, 1))
train_data = np.vstack((x1, x2)), np.vstack((y1, y2))


x_test = np.random.normal(size = (1000, 10))
y_test = x_test @ (beta - delta) + 0.1 * np.random.normal(size = (1000, 1))
test_minority = x_test, y_test

x_test = np.random.normal(size = (1000, 10))
y_test = x_test @ (beta + delta) + 0.1 * np.random.normal(size = (1000, 1))
test_majority = x_test, y_test

#nodes = int(float(sys.argv[1]))
for nodes in args.nodes:
    train_mse, majority_mse, minority_mse = mse_overparameter(train_data, test_majority, test_minority, nodes=nodes, optimizer='Adam')
    print(f'# Hidden nodes: {nodes}\nTrain mse: {train_mse}\nTest mse for majority: {majority_mse}\nTest mse for minority group: {minority_mse}\n\n')

