
import numpy as np
import matplotlib.pyplot as plt
import warnings
import termcolor
from My_Neural_Network import Activation_Functions
from My_Neural_Network.Obj_array import Obj_array
from My_Neural_Network import Initializer_Weights
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore',category=RuntimeWarning)

# noinspection PyTypeChecker
class Neural_Network:

    def __set_activation_functions(self, function):
        if (function == 'relu'): return tuple((Activation_Functions.relu, Activation_Functions.derivative_relu))
        if (function == 'leaky_relu'): return tuple((Activation_Functions.leaky_relu, Activation_Functions.derivative_leaky_relu))
        elif (function == 'identity'): return tuple((Activation_Functions.identity, Activation_Functions.derivative_identity))
        elif (function == 'tanh'): return tuple((Activation_Functions.tanh, Activation_Functions.derivative_tanh))
        elif (function == 'sigmoid'): return tuple((Activation_Functions.sigmoid, Activation_Functions.derivative_sigmoid))
        else: return None

    def __set_weights_initializer(self, weights_initializer):
        if(weights_initializer == 'Glorot_Uniform'): return Initializer_Weights.Glorot_Uniform_Initializer
        elif(weights_initializer == 'Glorot_Normal'): return Initializer_Weights.Glorot_Normal_Initializer
        elif(weights_initializer == 'Random_Normal'): return Initializer_Weights.random_normal
        elif(weights_initializer == 'Random_Uniform'): return Initializer_Weights.random_uniform
        elif(weights_initializer == 'Zeros'): return Initializer_Weights.Zeros
        elif(weights_initializer == 'Ones'): return Initializer_Weights.Ones
        else: return None

    def __init__(self, hidden_layer_sizes=(10,), epochs=200, tol=0.0001, no_iter_no_change=5, learning_rate_init=0.001, solver='SGD', decay=0, decay_adaptive=False, activation='relu', batch_size=32, regularizer=None, lambda_1=10e-4, lambda_2=10e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weights_initializer='Glorot_Uniform', verbose=False, shuffle=True, random_state=None):
        try:
            int(random_state)
            np.random.seed(random_state)
        except ValueError:
            print("Invalid seed,the default seed will be used")
        except TypeError:
            print("Invalid seed,the default seed will be used")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.tol = tol
        self.verbose = verbose
        self.learning_rate_init = learning_rate_init
        self.solver = solver
        self.decay = decay
        self.decay_adaptive = decay_adaptive
        self.__activation = self.__set_activation_functions(activation)
        self.activation_function = activation
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.shuffle = shuffle
        self.random_state = random_state
        self.__weights_initializer = self.__set_weights_initializer(weights_initializer)
        self.no_iter_no_change = no_iter_no_change
        self.loss_story = np.array([])
        self.__num_hidden_neurons = np.sum(hidden_layer_sizes)

    def count_right(self, y_true, pred):
        if (len(y_true) != len(pred)):
            print("len(y_true)!=len(y_pred), error!")
            return
        right = 0
        for i in range(len(y_true)):
            if (y_true[i] == pred[i]): right += 1
        return right

    def accuracy_score(self, y_true, y_pred): return (self.count_right(y_true, y_pred) / len(y_true))

    def __fixed_pred_proba(self, pred_proba):
        fixed_pred_proba = np.zeros((len(pred_proba), 2))
        for i in range(len(pred_proba)):
            fixed_pred_proba[i, 0] = (1 - pred_proba[i])
            fixed_pred_proba[i, 1] = pred_proba[i]
        return fixed_pred_proba

    def cross_entropy_loss(self, y, out_y): return -np.mean(y * np.log(out_y.T + self.epsilon))

    def loss_graphs(self):
        plt.title("Loss story")
        plt.plot(np.arange(0,self.last_epoch,1),self.loss_story)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    def __regularization(self, weights):
        if(self.regularizer=='l2'): return (self.lambda_2 * np.sum(weights**2))
        elif(self.regularizer=='l1'): return (self.lambda_1 * np.sum(np.abs(weights)))
        elif(self.regularizer=='l1_l2'): return (self.lambda_1 * np.sum(np.abs(weights)) + self.lambda_2 * np.sum(weights**2))
        else: return 0

    def __init_weights(self, input_features, output_layer):
        self.__W1 = np.array([],dtype=object); self.__B1 = np.array([],dtype=object)

        weight = Obj_array(self.__weights_initializer(input_features, self.hidden_layer_sizes[0], 0))
        bias = Obj_array(np.zeros(self.hidden_layer_sizes[0]))
        self.__W1 = np.append(self.__W1,weight)
        self.__B1 = np.append(self.__B1,bias)

        for i in range(1,len(self.hidden_layer_sizes)):
            weight = Obj_array(self.__weights_initializer(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i], i))
            bias = Obj_array(np.zeros(self.hidden_layer_sizes[i]))
            self.__W1 = np.append(self.__W1,weight)
            self.__B1 = np.append(self.__B1,bias)

        self.__W2 = self.__weights_initializer(self.hidden_layer_sizes[-1], output_layer, len(self.hidden_layer_sizes))
        self.__B2 = np.zeros((output_layer, 1))

    def __forward_propagation(self, X):
        weighted_X = np.array([],dtype=object); activate_X = np.array([],dtype=object)

        weight_sum = Obj_array(np.dot(X, self.__W1[0].np_array)+self.__B1[0].np_array)
        weighted_X = np.append(weighted_X,weight_sum)
        activate = Obj_array(self.__activation[0](weighted_X[0].np_array) + self.__regularization(self.__W1[0].np_array))
        activate_X = np.append(activate_X,activate)

        for i in range(1,len(self.hidden_layer_sizes)):
            weight_sum = Obj_array(np.dot(activate_X[i-1].np_array, self.__W1[i].np_array)+self.__B1[i].np_array)
            weighted_X = np.append(weighted_X,weight_sum)
            activate = Obj_array(self.__activation[0](weighted_X[i].np_array) + self.__regularization(self.__W1[i].np_array))
            activate_X = np.append(activate_X,activate)

        weighted_X_out = np.dot(self.__W2.T, activate_X[-1].np_array.T) + self.__B2
        activate_X_out = Activation_Functions.softmax(weighted_X_out)

        self.__save = tuple((weighted_X, activate_X, weighted_X_out, activate_X_out))
        return activate_X_out

    def __update_weights_SGD(self, dW_out, dB_out, dW1, dB1, epoch):
        dW1 = np.flip(dW1); dB1 = np.flip(dB1)
        for i in range(len(dW1)):
            self.__previous_momentum[i].np_array = (self.beta_1 * self.__previous_momentum[i].np_array) + self.learning_rate_init * dW1[i].np_array
            self.__W1[i].np_array -= self.__previous_momentum[i].np_array
            self.__B1[i].np_array -= self.learning_rate_init * dB1[i].np_array

        self.__previous_momentum[-1].np_array = (self.beta_1* self.__previous_momentum[-1].np_array) + self.learning_rate_init * dW_out.T
        self.__W2 -= self.__previous_momentum[-1].np_array
        self.__B2 -= self.learning_rate_init * dB_out

        self.learning_rate_init = self.learning_rate_init / (1 + self.decay * epoch)

    def __update_weights_ADAM(self, dW_out, dB_out, dW1, dB1, epoch):
        dW1 = np.flip(dW1); dB1 = np.flip(dB1)
        for i in range(len(dW1)):
            momentum_w = ((self.beta_1 * self.__previous_momentum[i].np_array) + (1 - self.beta_1) * dW1[i].np_array) / (1-np.power(self.beta_1, epoch))
            RMSprop_w = ((self.beta_2 * self.__previous_rms[i].np_array) + (1 - self.beta_2) * (dW1[i].np_array ** 2)) / (1-np.power(self.beta_2, epoch))
            self.__W1[i].np_array -= (self.learning_rate_init/(np.sqrt(RMSprop_w+self.epsilon))) * momentum_w

            momentum_b = ((self.beta_1 * self.__previous_momentum[i].np_array) + (1 - self.beta_1) * dB1[i].np_array) / (1-np.power(self.beta_1, epoch))
            RMSprop_b = ((self.beta_2 * self.__previous_rms[i].np_array) + (1 - self.beta_2) * (dB1[i].np_array ** 2)) / (1-np.power(self.beta_2, epoch))
            self.__B1[i].np_array -= (self.learning_rate_init/(np.sqrt(RMSprop_b+self.epsilon))) * momentum_b

        momentum_w = ((self.beta_1 * self.__previous_momentum[-1].np_array) + (1 - self.beta_1) * dW_out.T) / (1-np.power(self.beta_1, epoch))
        RMSprop_w = (self.beta_2 * self.__previous_rms[-1].np_array) + (1 - self.beta_2) * (dW_out.T ** 2) / (1 - np.power(self.beta_2, epoch))
        self.__W2 -= (self.learning_rate_init/(np.sqrt(RMSprop_w+self.epsilon))) * momentum_w

        momentum_b = ((self.beta_1 * self.__previous_momentum[-1].np_array) + (1 - self.beta_1) * dB_out) / (1 - np.power(self.beta_1, epoch))
        RMSprop_b = ((self.beta_2 * self.__previous_rms[-1].np_array) + (1 - self.beta_2) * (dB_out ** 2)) / (1 - np.power(self.beta_2, epoch))
        self.__B2 -= (self.learning_rate_init/(np.sqrt(RMSprop_b+self.epsilon))) * momentum_b

        self.learning_rate_init = self.learning_rate_init / (1 + self.decay * epoch)

    def __back_propagation(self, X, y):
        weighted_X1, activate_X1, weighted_X2, activate_X2 = self.__save

        # output_layer
        err_out = (activate_X2 - y.T)
        dW_out = np.dot(err_out, activate_X1[-1].np_array) / self.__num_hidden_neurons
        dB_out = np.sum(err_out) / self.__num_hidden_neurons

        # hidden_layer
        dW1 = np.array([],dtype=object); dB1 = np.array([],dtype=object)
        err_hidden_layer = None
        if(len(self.hidden_layer_sizes)>1):
            err_hidden_layer = np.dot(self.__W2, err_out) * self.__activation[1](weighted_X1[-1].np_array.T)
            dW1 = np.append(dW1,Obj_array(np.dot(activate_X1[-2].np_array.T, err_hidden_layer.T) / self.__num_hidden_neurons))
            dB1 = np.append(dB1,Obj_array(np.sum(err_hidden_layer) / self.__num_hidden_neurons))

        i = len(self.hidden_layer_sizes)
        while(i>2):
            err_hidden_layer = np.dot(self.__W1[i-1].np_array, err_hidden_layer) * self.__activation[1](weighted_X1[i-2].np_array.T)
            dW1 = np.append(dW1,Obj_array(np.dot(activate_X1[i-3].np_array.T, err_hidden_layer.T) / self.__num_hidden_neurons))
            dB1 = np.append(dB1,Obj_array(np.sum(err_hidden_layer) / self.__num_hidden_neurons))
            i -= 1

        if(len(self.hidden_layer_sizes)>1): err_hidden_layer = np.dot(self.__W1[1].np_array, err_hidden_layer) * self.__activation[1](weighted_X1[0].np_array.T)
        else: err_hidden_layer = np.dot(self.__W2, err_out) * self.__activation[1](weighted_X1[0].np_array.T)
        dW1 = np.append(dW1,Obj_array(np.dot(X.T, err_hidden_layer.T) / self.__num_hidden_neurons))
        dB1 = np.append(dB1,Obj_array(np.sum(err_hidden_layer) / self.__num_hidden_neurons))

        return dW_out, dB_out, dW1, dB1

    def __set_sigmoid_output(self, out_X):
        fixed_X = np.array([])
        for i in range(len(out_X)):
            fixed_X = np.append(fixed_X, (1 if out_X[i] >= 0.5 else 0))
        return fixed_X

    def __set_encoder(self,y):
        self.__enc = OneHotEncoder(sparse=False, categories='auto')
        self.__enc.fit(y.reshape(len(y), -1))

    def __encoding(self,y):
        y = self.__enc.transform(y.reshape(len(y), -1))
        return y

    def __shuffle(self, X, y):
        X_shuffle = np.zeros(X.shape); y_shuffle = np.zeros(y.shape)
        arr_index = np.arange(0, X.shape[0], 1)
        for i in range(X_shuffle.shape[0]):
            index = np.random.choice(arr_index)
            filter_array = arr_index != index
            arr_index = arr_index[filter_array]

            X_shuffle[i] = X[index]
            y_shuffle[i] = y[index]

        return X_shuffle, y_shuffle

    def fit(self, X, y):
        if (self.__activation == None):
            self.activation_function = None
            print("Invalid activation functions")
            return
        if (self.batch_size<=0):
            print("Invalid batch_size")
            return

        self.__init_weights(X.shape[1], len(np.unique(y)))

        self.__set_encoder(y)
        y = self.__encoding(y)

        previous_loss = None; iter_change = 0; loss = None

        self.__previous_momentum = np.array([])
        self.__previous_rms = np.array([])
        for _ in range(len(self.hidden_layer_sizes)+1):
            self.__previous_momentum = np.append(self.__previous_momentum,Obj_array(0))
            if(self.solver=='ADAM'): self.__previous_rms = np.append(self.__previous_rms, Obj_array(0))

        for epoch in range(self.epochs):
            batch = 0
            if(self.shuffle): X, y = self.__shuffle(X, y)
            while(True):
                X_batch = X[batch*self.batch_size:(batch+1)*self.batch_size]
                y_batch = y[batch*self.batch_size:(batch+1)*self.batch_size]

                if(X_batch.shape[0]==0): break

                out_y = self.__forward_propagation(X_batch)

                dW_out, dB_out, dW1, dB1 = self.__back_propagation(X_batch, y_batch)
                if(self.solver=='ADAM'): self.__update_weights_ADAM(dW_out, dB_out, dW1, dB1, epoch=epoch+1)
                else: self.__update_weights_SGD(dW_out, dB_out, dW1, dB1, epoch=epoch+1)

                loss = self.cross_entropy_loss(y_batch,out_y)
                if(previous_loss!=None):
                    if(loss>(previous_loss-self.tol)): iter_change +=1
                    else: iter_change = 0
                previous_loss = loss
                if(self.decay_adaptive and iter_change==2): self.learning_rate_init /=5
                if(self.no_iter_no_change==iter_change):
                    self.loss_story = np.append(self.loss_story, loss)
                    self.last_epoch = epoch + 1
                    print(termcolor.colored(f"Training loss did not improve more than tol={self.tol} for {self.no_iter_no_change} consecutive step of gradient descent. Stopping.",'red'))
                    return
                batch += 1

            self.loss_story = np.append(self.loss_story, loss)
            if(self.verbose): print(f"In epoch {epoch+1}, log loss = {loss}")
            self.last_epoch = epoch + 1

        if(np.isnan(loss)): print(termcolor.colored("Please resize the data",'red'))

    def predict(self, X):
        y = self.__forward_propagation(X)
        return np.argmax(y,axis=0)

    def predict_proba(self, X):
        y = self.__forward_propagation(X)
        return y

    def show_results(self, y_true, y_pred, pred_proba, show_loss_graphs=False):
        __y_true = self.__encoding(y_true)
        print("Log loss = ", self.cross_entropy_loss(__y_true, pred_proba))
        print("Accuracy score = ", self.accuracy_score(np.argmax(__y_true,axis=1), y_pred))
        print(f"{self.count_right(np.argmax(__y_true,axis=1), y_pred)} correct answers out of {__y_true.shape[0]}")
        if(show_loss_graphs): self.loss_graphs()
