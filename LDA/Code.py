import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from starter import *
from sklearn.preprocessing import PolynomialFeatures 


######################################################
######## Part A: Models used for predictions. ########
######################################################

def compute_update(single_obj_loc, sensor_loc, single_distance):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    grad = loc_difference / np.expand_dims(phi, 1)  # k * 2.
    update = np.linalg.solve(grad.T.dot(grad), grad.T.dot(single_distance - phi))

    return update


def get_object_location(sensor_loc, single_distance, num_iters=20, num_repeats=10):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """
    obj_locs = np.zeros((num_repeats, 1, 2))
    distances = np.zeros(num_repeats)
    for i in range(num_repeats):
        obj_loc = np.random.randn(1, 2) * 100
        for t in range(num_iters):
            obj_loc += compute_update(obj_loc, sensor_loc, single_distance)

        distances[i] = np.sum((single_distance - np.linalg.norm(obj_loc - sensor_loc, axis=1))**2)
        obj_locs[i] = obj_loc

    obj_loc = obj_locs[np.argmin(distances)]

    return obj_loc[0]


def generative_model(X, Y, Xs_test, Ys_test):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    initial_sensor_loc = np.random.randn(7, 2) * 100
    estimated_sensor_loc = find_mle_by_grad_descent_part_e(
        initial_sensor_loc, Y, X, lr=0.001, num_iters=1000)

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(
            [get_object_location(estimated_sensor_loc, X_test_single) for X_test_single in X_test])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def oracle_model(X, Y, Xs_test, Ys_test, sensor_loc):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    sensor_loc: location of the sensors.
    Output:
    mse: Mean square error on test data.
    """
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array([
            get_object_location(sensor_loc, X_test_single)
            for X_test_single in X_test
        ])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def construct_second_order_data(X):
    """
    This function computes second order variables 
    for polynomial regression.
    Input:
    X: Independent variables.
    Output:
    A data matrix composed of both first and second order terms. 
    """
    X_second_order = []
    m = X.shape[1]
    for i in range(m):
        for j in range(m):
            if j <= i:
                X_second_order.append(X[:,i] * X[:,j])
    X_second_order = np.array(X_second_order).T
    return np.concatenate((X,X_second_order), axis = 1)


def linear_regression(X, Y, Xs_test, Ys_test):
    """
    This function performs linear regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """

    X_n = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    XL = np.concatenate((X_n, np.ones((len(X),1))), axis = 1)
    w = np.linalg.solve(XL.T.dot(XL),XL.T.dot(Y))
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        XL_test = np.concatenate(((X_test - np.mean(X, axis = 0)) / np.std(X, axis = 0), 
                                  np.ones((len(X_test),1))), axis = 1)
        Y_pred = XL_test.dot(w)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis = 1))) 
        mses.append(mse)  
    return mses


def poly_regression_second(X, Y, Xs_test, Ys_test):
    """
    This function performs second order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    
    poly = PolynomialFeatures(degree = 2)
    X2 = poly.fit_transform(X)[:,1:]
    X2_test = []
    for X_test in Xs_test:
        X2_test.append(poly.fit_transform(X_test)[:,1:])
    mses = linear_regression(X2, Y, X2_test, Ys_test)
    return mses


def poly_regression_cubic(X, Y, Xs_test, Ys_test):
    """
    This function performs third order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    
    poly = PolynomialFeatures(degree = 3)
    X3 = poly.fit_transform(X)[:,1:]
    X3_test = []
    for X_test in Xs_test:
        X3_test.append(poly.fit_transform(X_test)[:,1:])
    mses = linear_regression(X3, Y, X3_test, Ys_test)
    return mses


def neural_network(X, Y, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
     
    X_n = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    Y_n = (Y - np.mean(Y, axis = 0)) / np.std(Y, axis = 0)
    
    model = Model(X_n.shape[1])
    model.addLayer(DenseLayer(100, ReLUActivation()))
    model.addLayer(DenseLayer(100, ReLUActivation()))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = model.predict((X_test - np.mean(X, axis = 0)) / np.std(X, axis = 0)) 
        Y_pred = Y_pred * np.std(Y, axis = 0) + np.mean(Y, axis = 0)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis = 1)) )   
        mses.append(mse)

    return mses 


######################################################
####################### Part B #######################
######################################################

def main():
    np.random.seed(0)
    sensor_loc = generate_sensors()
    regular_loc, _ = generate_dataset(
        sensor_loc,
        num_sensors=sensor_loc.shape[0],
        spatial_dim=2,
        num_data=20,
        original_dist=True,
        noise=1)
    shifted_loc, _ = generate_dataset(
        sensor_loc,
        num_sensors=sensor_loc.shape[0],
        spatial_dim=2,
        num_data=20,
        original_dist=False,
        noise=1)

    plt.scatter(sensor_loc[:, 0], sensor_loc[:, 1], label="sensors")
    plt.scatter(regular_loc[:, 0], regular_loc[:, 1], label="regular points")
    plt.scatter(shifted_loc[:, 0], shifted_loc[:, 1], label="shifted points")
    plt.legend()
    plt.savefig("dataset.png")
    plt.show()

if __name__ == '__main__':
    main()     


######################################################
####################### Part C #######################
######################################################

def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)
    #ns = np.concatenate((np.arange(10,110,10), np.arange(200,1100,100)),axis = 0)
    # ns = np.arange(100,550,50)#np.arange(10,110,10)
    ns = np.arange(10,310,20)
    replicates = 5 
    num_methods = 5
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))
    def generate_data(sensor_loc, k = 7, d = 2, 
                 n = 1, original_dist = True, noise = 1): 
        return generate_dataset(sensor_loc, num_sensors = k, spatial_dim = d, 
                     num_data = n, original_dist = original_dist, noise = noise)
    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n = 1000)
        X_test2, Y_test2 = generate_data(sensor_loc, n = 1000, original_dist = False)
        for t,n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n = n) # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            ### Linear regression: 
            mse = linear_regression(X,Y,Xs_test,Ys_test)
            mses[t, s, 0] = mse
            
            ### Second-order Polynomial regression: 
            mse = poly_regression_second(X,Y, Xs_test, Ys_test)
            mses[t, s, 1] = mse
            
            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test)
            mses[t, s, 2] = mse

            ### Generative model:
            mse = generative_model(X,Y,Xs_test,Ys_test)
            mses[t, s, 3] = mse

            ### Second-order Polynomial regression: 
            mse = poly_regression_cubic(X,Y, Xs_test, Ys_test)
            mses[t, s, 4] = mse
        
            print('{}th Experiment with {} samples done...'.format(s, n))
            
    ### Plot MSE for each model. 
    plt.figure()
    regressors = ['Linear Regression','2nd-order Polynomial Regression', 
    'Neural Network', 'Generative Model','3rd-order Polynomial Regression']
    for a in range(5):
        plt.plot(ns, np.mean(mses[:,:,a, 0], axis = 1), label = regressors[a])
        
    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    # plt.show()    

    plt.figure()
    regressors = ['Linear Regression','2nd-order Polynomial Regression', 
    'Neural Network', 'Generative Model','3rd-order Polynomial Regression']
    for a in range(5):
        plt.plot(ns, np.mean(mses[:,:,a, 1], axis = 1), label = regressors[a])
        
    plt.title('Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    # plt.show()    

    plt.figure()
    regressors = ['Linear Regression','2nd-order Polynomial Regression', 
    'Neural Network', 'Generative Model','3rd-order Polynomial Regression']
    for a in range(5):
        plt.plot(ns, np.mean(mses[:,:,a, 2], axis = 1), label = regressors[a])
        
    plt.title('Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.savefig('val_different_mse.png')
    plt.show()    
 

if __name__ == '__main__':
    main()               


######################################################
####################### Part D #######################
######################################################

def neural_network(X, Y, X_test, Y_test, num_neurons, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_neurons: number of neurons in each layer
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    
    X_n = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    Y_n = (Y - np.mean(Y, axis = 0)) / np.std(Y, axis = 0)
    
    if activation == "ReLU":
        activation = ReLUActivation
    elif activation == "tanh":
        activation = TanhActivation
    else:
        print('Null activation')
    
    model = Model(X_n.shape[1])
    model.addLayer(DenseLayer(num_neurons, activation()))
    model.addLayer(DenseLayer(num_neurons, activation()))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())

    Y_pred = model.predict((X_test - np.mean(X, axis = 0)) / np.std(X, axis = 0)) 
    Y_pred = Y_pred * np.std(Y, axis = 0) + np.mean(Y, axis = 0)
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis = 1)))   

    return mse

#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_neuronss = np.arange(100, 550, 50)
mses = np.zeros((len(num_neuronss), 2))

# for s in range(replicates):

sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_neurons in enumerate(num_neuronss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} neurons done...'.format(num_neurons))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_neuronss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of neurons')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
#plt.savefig('num_neurons.png')
plt.show()



######################################################
####################### Part E #######################
######################################################

def neural_network(X, Y, X_test, Y_test, num_layers, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    
    X_n = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    Y_n = (Y - np.mean(Y, axis = 0)) / np.std(Y, axis = 0)
    
    if num_layers == 1:
        num_neurons = 1000
    elif num_layers == 2:
        num_neurons = 200
    elif num_layers == 3:
        num_neurons = 141
    elif num_layers == 4:
        num_neurons = 115

    if activation == "ReLU":
        activation = ReLUActivation
    elif activation == "tanh":
        activation = TanhActivation
    else:
        print('Null activation')

    model = Model(X_n.shape[1])
    model.addLayer(DenseLayer(num_neurons,activation()))
    if num_layers >= 2:
        model.addLayer(DenseLayer(num_neurons,activation()))
    if num_layers >= 3:
        model.addLayer(DenseLayer(num_neurons,activation()))

    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())
    
    Y_pred = model.predict((X_test - np.mean(X, axis = 0)) / np.std(X, axis = 0)) 
    Y_pred = Y_pred * np.std(Y, axis = 0) + np.mean(Y, axis = 0)
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis = 1)))      
    return mse 


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_layerss = [1, 2, 3, 4]
mses = np.zeros((len(num_layerss), 2))

# for s in range(replicates):
sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_layers in enumerate(num_layerss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_layers, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_layers, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} layers done...'.format(num_layers))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_layerss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of layers')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
#plt.savefig('num_layers.png')
plt.show()


######################################################
####################### Part F #######################
######################################################

def neural_network(X, Y, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """

    X_n = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    Y_n = (Y - np.mean(Y, axis = 0)) / np.std(Y, axis = 0)
    
    model = Model(X_n.shape[1])
    model.addLayer(DenseLayer(100, ReLUActivation()))
    model.addLayer(DenseLayer(100, ReLUActivation()))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = model.predict((X_test - np.mean(X, axis = 0)) / np.std(X, axis = 0)) 
        Y_pred = Y_pred * np.std(Y, axis = 0) + np.mean(Y, axis = 0)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis = 1)) )   
        mses.append(mse)

    return mses 
    

def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(10, 310, 20)
    replicates = 5
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            ### Linear regression:
            mse = linear_regression(X, Y, Xs_test, Ys_test)
            mses[t, s, 0] = mse

            ### Second-order Polynomial regression:
            mse = poly_regression_second(X, Y, Xs_test, Ys_test)
            mses[t, s, 1] = mse

            ### 3rd-order Polynomial regression:
            mse = poly_regression_cubic(X, Y, Xs_test, Ys_test)
            mses[t, s, 2] = mse

            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test)
            mses[t, s, 3] = mse

            ### Generative model:
            mse = generative_model(X, Y, Xs_test, Ys_test)
            mses[t, s, 4] = mse

            ### Oracle model:
            mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))

    ### Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    #plt.savefig('val_different_mse.png')
    plt.show()


if __name__ == '__main__':
    main()