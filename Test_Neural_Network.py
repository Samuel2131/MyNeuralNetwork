
from Neural_Network import Neural_Network
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def create_dataset_cl():
    data = load_iris()['data']
    target = load_iris()['target']
    return tuple((data,target))

def scaling(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X

def test_model(data,target,test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42)

    model = Neural_Network(hidden_layer_sizes=(100, ), learning_rate_init=0.01, solver='ADAM', batch_size=16, epochs=500, tol=0.0001, no_iter_no_change=10, verbose=True, random_state=42)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_proba_train = model.predict_proba(X_train)
    pred_proba_test = model.predict_proba(X_test)

    print("Activation = ",model.activation_function)
    print("Batch size = ",model.batch_size)
    print("Regularization = ",model.regularizer)
    print("Results train set = ")
    model.show_results(y_train, pred_train, pred_proba_train, show_loss_graphs=False)
    print("Results test set = ")
    model.show_results(y_test, pred_test, pred_proba_test, show_loss_graphs=True)

def main():
    data, target = create_dataset_cl()
    data = scaling(data)
    test_model(data,target, test_size=0.2)

if(__name__=='__main__'):
    main()

