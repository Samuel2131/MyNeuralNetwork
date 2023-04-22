
import numpy as np

def random_normal(input_features, hidden_layer_size, *args):
    return np.random.normal(loc=0, scale=0.05, size=(input_features, hidden_layer_size))

def random_uniform(input_features, hidden_layer_size, *args):
    return np.random.uniform(low=-0.05, high=0.05, size=(input_features, hidden_layer_size))

def Zeros(input_features, hidden_layer_size, *args):
    return np.zeros(shape=(input_features, hidden_layer_size))

def Ones(input_features, hidden_layer_size, *args):
    return np.ones(shape=(input_features, hidden_layer_size))

def Glorot_Uniform_Initializer(input_features, hidden_layer_size, num_layer):
    num_layer+=1
    MAX = np.sqrt(2 / (np.power(hidden_layer_size, num_layer - 1) + np.power(hidden_layer_size, num_layer)))
    MIN = -(np.sqrt(2 / (np.power(hidden_layer_size, num_layer - 1) + np.power(hidden_layer_size, num_layer))))

    matrix_weights = np.random.uniform(low=MIN, high=MAX, size=(input_features,hidden_layer_size))

    return matrix_weights

def Glorot_Normal_Initializer(input_features, hidden_layer_size, num_layer):
    num_layer+=1
    MAX = 1; MIN = -1
    sqrt = np.sqrt(2 / (np.power(hidden_layer_size, num_layer - 1) + np.power(hidden_layer_size, num_layer)))

    matrix_weights = (np.random.uniform(low=MIN, high=MAX, size=(input_features,hidden_layer_size)) * sqrt)

    return matrix_weights
