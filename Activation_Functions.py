
import numpy as np

def identity(X):
    return X

def derivative_identity(X):
    return np.ones(X.shape)

def relu(X):
    return np.maximum(X, 0)

def leaky_relu(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = X[i, j] if X[i, j] >= 0 else 0.1 * X[i, j]
    return X

def derivative_leaky_relu(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = 1 if X[i, j] >= 0 else 0.01
    return X

def derivative_relu(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
             X[i, j] = 1 if X[i, j] > 0 else 0
    return X

def tanh(X):
    return (np.power(np.e,X)-np.power(np.e,-X))/(np.power(np.e,X)+np.power(np.e,-X))

def derivative_tanh(X):
    return 1 - (np.power((np.power(np.e, X) - np.power(np.e, -X)), 2) / np.power((np.power(np.e, X) + np.power(np.e, -X)), 2))

def sigmoid(X):
    return 1 / (1 + np.power(np.e, -X))

def derivative_sigmoid(X):
    return np.power(np.e, -X) / np.power((np.power(np.e, -X) + 1), 2)

def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=0, keepdims=True)
