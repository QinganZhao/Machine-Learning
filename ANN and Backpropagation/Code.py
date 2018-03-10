###############################
######### Problem 2 ###########
###############################

""" Tools for calculating Gradient Descent for ||Ax-b||. """
import matplotlib.pyplot as plt
import numpy as np


def main():
    ################################################################################
    # Input Variables
    A = np.array([[15, 8], [6, 5]])  # do not change this until the last part
    b = np.array([4.5, 6])  # b in the equation ||Ax-b||
    initial_position = np.array([0, 0])  # position at iteration 0
    total_step_count = 200  # number of GD steps to take
    step_size = lambda i: 1/(1+i)  # step size at iteration i
    ################################################################################

    # computes desired number of steps of gradient descent
    positions = compute_updates(A, b, initial_position, total_step_count, step_size)

    # print out the values of the x_i
    print(positions)
    print(np.dot(np.linalg.inv(A), b))

    # plot the values of the x_i
    plt.scatter(positions[:, 0], positions[:, 1], c='blue')
    plt.scatter(np.dot(np.linalg.inv(A), b)[0],
                np.dot(np.linalg.inv(A), b)[1], c='red')
    plt.plot()
    plt.show()


def compute_gradient(A, b, x):
    """Computes the gradient of ||Ax-b|| with respect to x."""
    return np.dot(A.T, (np.dot(A, x) - b)) / np.linalg.norm(np.dot(A, x) - b)


def compute_update(A, b, x, step_count, step_size):
    """Computes the new point after the update at x."""
    return x - step_size(step_count) * compute_gradient(A, b, x)


def compute_updates(A, b, p, total_step_count, step_size):
    """Computes several updates towards the minimum of ||Ax-b|| from p.

    Params:
        b: in the equation ||Ax-b||
        p: initialization point
        total_step_count: number of iterations to calculate
        step_size: function for determining the step size at step i
    """
    positions = [np.array(p)]
    for k in range(total_step_count):
        positions.append(compute_update(A, b, positions[-1], k, step_size))
    return np.array(positions)


main()


###############################
######### Problem 4 ###########
###############################

from common import *

###############################
######### Part b ##############
###############################

########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc, sensor_loc, single_distance):
    """
    Compute the gradient of the loglikelihood function for part a.   

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
    loc_diff = single_obj_loc - sensor_loc
    tmp = np.linalg.norm(loc_diff, axis=1) 
    weight = (tmp - single_distance) / tmp 
    grad = -np.sum(np.expand_dims(weight, axis=1) * loc_diff, axis=0)

    return grad

def find_mle_by_grad_descent_part_b(initial_obj_loc, sensor_loc, single_distance, lr=0.001, num_iters = 10000):
    """
    Compute the gradient of the loglikelihood function for part a.   

    Input:
    initial_obj_loc: 1 * d numpy array. 
    Initialized Location of the single object.

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array. 
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """    
    obj_loc = initial_obj_loc
    for i in range(num_iters):
        obj_loc = obj_loc + lr * compute_gradient_of_likelihood(obj_loc, sensor_loc, single_distance)

    return obj_loc

if __name__ == "__main__":
    ########################################################################
    #########  MAIN ########################################################
    ########################################################################

    # Your code: set some appropriate learning rate here
    lr = 0.01

    np.random.seed(0)
    sensor_loc = generate_sensors()
    obj_loc, distance = generate_data(sensor_loc)
    single_distance = distance[0]
    print('The real object location is')
    print(obj_loc)
    # Initialized as [0,0]
    initial_obj_loc = np.array([[0.,0.]]) 
    estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
               sensor_loc, single_distance, lr=lr, num_iters = 10000)
    print('The estimated object location with zero initialization is')
    print(estimated_obj_loc)

    # Random initialization.
    initial_obj_loc = np.random.randn(1,2)
    estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
               sensor_loc, single_distance, lr=lr, num_iters = 10000)
    print('The estimated object location with random initialization is')
    print(estimated_obj_loc)   


from common import *
#from part_b_starter import find_mle_by_grad_descent_part_b

#############################
######### Part c ############
#############################
def log_likelihood(obj_loc, sensor_loc, distance): 
    """
    This function computes the log likelihood (as expressed in Part a).
    Input: 
    obj_loc: shape [1,2]
    sensor_loc: shape [7,2]
    distance: shape [7]
    Output: 
    The log likelihood function value. 
    """  
    diff = np.sqrt(np.sum((sensor_loc - obj_loc)**2, axis = 1)) - distance
    func_value = -sum((diff) ** 2) / 2
    return func_value

if __name__ == "__main__":
    ################################################################################
    ######### Compute the function value at local minimum for all experiments. #####
    ################################################################################
    num_sensors = 20

    np.random.seed(100)
    sensor_loc = generate_sensors(k=num_sensors)

    # num_data_replicates = 10
    num_gd_replicates = 100

    obj_locs = [[[i,i]] for i in np.arange(0,1000,100)]

    func_values = np.zeros((len(obj_locs),10, num_gd_replicates))
    # record sensor_loc, obj_loc, 100 found minimas
    minimas = np.zeros((len(obj_locs), 10, num_gd_replicates, 2))
    true_object_locs = np.zeros((len(obj_locs), 10, 2))

    for i, obj_loc in enumerate(obj_locs): 
        for j in range(10):
            obj_loc, distance = generate_data_given_location(sensor_loc, obj_loc, 
                                                           k = num_sensors, d = 2)
            true_object_locs[i, j, :] = np.array(obj_loc)

            for gd_replicate in range(num_gd_replicates): 
                initial_obj_loc = np.random.randn(1,2) * (100 * i+1)
                obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
                         sensor_loc, distance[0], lr=0.1, num_iters = 1000) 
                minimas[i, j, gd_replicate, :] = np.array(obj_loc)
                func_value = log_likelihood(obj_loc, sensor_loc, distance[0])
                func_values[i, j, gd_replicate] = func_value

    ########################################################################
    ######### Calculate the things to be plotted. ##########################
    ########################################################################
    local_mins = [[np.unique(func_values[i,j].round(decimals=2)) for j in range(10)] for i in range(10)]
    num_local_min = [[len(local_mins[i][j]) for j in range(10)] for i in range(10)]
    proportion_global = [[sum(func_values[i,j].round(decimals=2) == min(local_mins[i][j]))*1.0/100 \
                         for j in range(10)] for i in range(10)]


    num_local_min = np.array(num_local_min)
    num_local_min = np.mean(num_local_min, axis = 1)

    proportion_global = np.array(proportion_global)
    proportion_global = np.mean(proportion_global, axis = 1)

    ########################################################################
    ######### Plots. #######################################################
    ########################################################################
    fig, axes = plt.subplots(figsize=(8,6), nrows=2, ncols=1)
    fig.tight_layout()
    plt.subplot(211)

    plt.plot(np.arange(0,1000,100), num_local_min)
    plt.title('Number of local minimum found by 100 gradient descents.')
    plt.xlabel('Object Location')
    plt.ylabel('Number')
    #plt.show()
    #plt.savefig('num_obj.png')
    # Proportion of gradient descents that find the local minimum of minimum value. 

    plt.subplot(212)
    plt.plot(np.arange(0,1000,100), proportion_global)
    plt.title('Proportion of GD that finds the global minimum among 100 gradient descents.')
    plt.xlabel('Object Location')
    plt.ylabel('Proportion')
    fig.tight_layout()
    #plt.show()
    #plt.savefig('prop_obj.png')

    ########################################################################
    ######### Plots of contours. ###########################################
    ########################################################################
    np.random.seed(0) 
    # sensor_loc = np.random.randn(7,2) * 10
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    X, Y = np.meshgrid(x, y) 
    obj_loc = [[0,0]]
    obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                   obj_loc, k = num_sensors, d = 2)

    Z =  np.array([[log_likelihood((X[i,j],Y[i,j]), 
                                 sensor_loc, distance[0]) for j in range(len(X))] \
                 for i in range(len(X))]) 


    plt.figure(figsize=(10,4))
    plt.subplot(121)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('With object at (0,0)')

    np.random.seed(0) 
    # sensor_loc = np.random.randn(7,2) * 10
    x = np.arange(-400,400, 4)
    y = np.arange(-400,400, 4)
    X, Y = np.meshgrid(x, y) 
    obj_loc = [[200,200]]
    obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                   obj_loc, k = num_sensors, d = 2)

    Z =  np.array([[log_likelihood((X[i,j],Y[i,j]), 
                                 sensor_loc, distance[0]) for j in range(len(X))] \
                 for i in range(len(X))]) 


    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    #plt.figure()
    plt.subplot(122)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('With object at (200,200)')
    #plt.savefig('likelihood_landscape.png')


    ########################################################################
    ######### Plots of Found local minimas. ###########################################
    ########################################################################
    #sensor_loc
    #minimas = np.zeros((len(obj_locs), 10, num_gd_replicates, 2))
    #true_object_locs = np.zeros((len(obj_locs), 10, 2))
    object_loc_i = 5
    trail = 0

    plt.figure()
    plt.plot(sensor_loc[:, 0], sensor_loc[:, 1], 'r+', label="sensors")
    plt.plot(minimas[object_loc_i, trail, :, 0], minimas[object_loc_i, trail, :, 1], 'g.', label="minimas")
    plt.plot(true_object_locs[object_loc_i, trail, 0], true_object_locs[object_loc_i, trail, 1], 'b*', label="object")
    plt.title('object at location (%d, %d), gradient descent recovered locations' % (object_loc_i*100, object_loc_i*100))
    plt.legend()
    plt.show()


###############################
######### Part f ##############
###############################

########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_grad_likelihood(sensor_loc, obj_loc, distance):
    """
    Compute the gradient of the loglikelihood function for part f.   

    Input:
    sensor_loc: k * d numpy array. 
    Location of sensors.

    obj_loc: n * d numpy array. 
    Location of the objects.

    distance: n * k dimensional numpy array. 
    Observed distance of the object.

    Output:
    grad: k * d numpy array.
    """
    grad = np.zeros(sensor_loc.shape)
    ### finish the grad loglike ###
    for i, j in enumerate(sensor_loc):
        d = distance[:, i] 
        grad[i] = compute_gradient_of_likelihood(j, obj_loc, d)

    return grad

def find_mle_by_grad_descent(initial_sensor_loc, 
           obj_loc, distance, lr=0.001, num_iters = 1000):
    """
    Compute the gradient of the loglikelihood function for part f.   

    Input:
    initial_sensor_loc: k * d numpy array. 
    Initialized Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    distance: n * k dimensional numpy array. 
    Observed distance of the n object.

    Output:
    sensor_loc: k * d numpy array. The mle for the location of the object.

    """    
    sensor_loc = initial_sensor_loc
    ### finish the gradient descent ###
    for i in range(num_iters):
        sensor_loc = sensor_loc + lr * compute_grad_likelihood(sensor_loc, obj_loc, distance) 

    return sensor_loc
########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################

np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc, n = 100)
print('The real sensor locations are')
print(sensor_loc)
# Initialized as zeros.
initial_sensor_loc = np.zeros((7,2)) #np.random.randn(7,2)
estimated_sensor_loc = find_mle_by_grad_descent(initial_sensor_loc,
                                                obj_loc, distance, lr=0.001, num_iters = 1000)
print('The predicted sensor locations are')
print(estimated_sensor_loc) 

 
########################################################################
#########  Estimate distance given estimated sensor locations. ######### 
########################################################################

def compute_distance_with_sensor_and_obj_loc(sensor_loc, obj_loc):
    """
    stimate distance given estimated sensor locations.  

    Input:
    sensor_loc: k * d numpy array. 
    Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    Output:
    distance: n * k dimensional numpy array. 
    """ 
    estimated_distance = scipy.spatial.distance.cdist(obj_loc, sensor_loc, metric='euclidean')
    return estimated_distance 

########################################################################
#########  MAIN  #######################################################
########################################################################    
np.random.seed(100)    
########################################################################
#########  Case 1. #####################################################
########################################################################
mse =0

for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = True)
    #obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    ### compute the mse for this case ###
    estimate = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, obj_loc)
    mse = mse + np.mean(np.sum(estimate, axis = 1))

mse = mse / 100
    
              
print('The MSE for Case 1 is {}'.format(mse))

########################################################################
#########  Case 2. #####################################################
########################################################################
mse = 0

for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = False)
    #obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    ### compute the mse for this case ###
    estimate = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, obj_loc)
    mse = mse + np.mean(np.sum(estimate, axis = 1))

mse = mse / 100

print('The MSE for Case 2 is {}'.format(mse)) 


########################################################################
#########  Case 3. #####################################################
########################################################################
mse =0

for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = False)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    # Your code: compute the mse for this case
    estimate = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, obj_loc)
    mse = mse + np.mean(np.sum(estimate, axis = 1))

mse = mse / 100

print('The MSE for Case 2 (if we knew mu is [300,300]) is {}'.format(mse)) 

###############################
######### Problem 5 ###########
###############################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Gradient descent optimization
# The learning rate is specified by eta
class GDOptimizer(object):
    def __init__(self, eta):
        self.eta = eta

    def initialize(self, layers):
        pass

    # This function performs one gradient descent step
    # layers is a list of dense layers in the network
    # g is a list of gradients going into each layer before the nonlinear activation
    # a is a list of of the activations of each node in the previous layer going
    #
    def update(self, layers, g, a):
        m = a[0].shape[1]
        for layer, curGrad, curA in zip(layers, g, a):
            
            ############################
            ########## PART F ##########
            ############################
            
            ########################################################################################
            # Compute the gradients for layer.W and layer.b using the gradient for the output of the
            # layer curA and the gradient of the output curGrad
            # Use the gradients to update the weight and the bias for the layer
            #
            # Normalize the learning rate by m (defined above), the number of training examples input
            # (in parallel) to the network.
            #
            # It may help to think about how you would calculate the update if we input just one
            # training example at a time; then compute a mean over these individual update values.
            # ######################################################################################
            
            WeightUpdate = curGrad @ curA.T
            BiasUpdate = np.sum(curGrad, axis=1).reshape(layer.b.shape)
            layer.updateWeights(-self.eta/m * WeightUpdate)
            layer.updateBias(-self.eta/m * BiasUpdate)

# Cost function used to compute prediction errors
class QuadraticCost(object):

    # Compute the squared error between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(y,yp):
        
        ############################
        ########## PART B ##########
        ############################
        
        return 0.5 * (y - yp) ** 2 

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(y,yp):
        
        ############################
        ########## PART B ##########
        ############################
        
        return yp - y

# Sigmoid function fully implemented as an example
class SigmoidActivation(object):
    @staticmethod
    def fx(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dx(z):
        return SigmoidActivation.fx(z) * (1 - SigmoidActivation.fx(z))

# Hyperbolic tangent function
class TanhActivation(object):

    # Compute tanh for each element in the input z
    @staticmethod
    def fx(z):
        
        ############################
        ########## PART C ##########
        ############################
        
        return np.tanh(z)

    # Compute the derivative of the tanh function with respect to z
    @staticmethod
    def dx(z):
        
        ############################
        ########## PART C ##########
        ############################
        
        return 1 - np.tanh(z) ** 2

# Rectified linear unit
class ReLUActivation(object):
    @staticmethod
    def fx(z):
        
        ############################
        ########## PART C ##########
        ############################
        
        return np.where(z < 0, 0, z)

    @staticmethod
    def dx(z):
        
        ############################
        ########## PART C ##########
        ############################
        
        return np.where(z <= 0, 0, 1)

# Linear activation
class LinearActivation(object):
    @staticmethod
    def fx(z):
        
        ############################
        ########## PART C ##########
        ############################
        
        return z

    @staticmethod
    def dx(z):
        
        ############################
        ########## PART C ##########
        ############################
        
        return np.ones(z.shape)

# This class represents a single hidden or output layer in the neural network
class DenseLayer(object):

    # numNodes: number of hidden units in the layer
    # activation: the activation function to use in this layer
    def __init__(self, numNodes, activation):
        self.numNodes = numNodes
        self.activation = activation

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        s = scale * np.sqrt(6.0 / (self.numNodes + fanIn))
        self.W = np.random.normal(0, s,
                                   (self.numNodes,fanIn))
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        return self.W.dot(a) + self.b # Note, this is implemented where we assume a is k x n

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        return self.activation.dx(z)

    # Update the weights of the layer by adding dW to the weights
    def updateWeights(self, dW):
        self.W = self.W + dW

    # Update the bias of the layer by adding db to the bias
    def updateBias(self, db):
        self.b = self.b + db

# This class handles stacking layers together to form the completed neural network
class Model(object):

    # inputSize: the dimension of the inputs that go into the network
    def __init__(self, inputSize):
        self.layers = []
        self.inputSize = inputSize

    # Add a layer to the end of the network
    def addLayer(self, layer):
        self.layers.append(layer)

    # Get the output size of the layer at the given index
    def getLayerSize(self, index):
        if index >= len(self.layers):
            return self.layers[-1].getNumNodes()
        elif index < 0:
            return self.inputSize
        else:
            return self.layers[index].getNumNodes()

    # Initialize the weights of all of the layers in the network and set the cost
    # function to use for optimization
    def initialize(self, cost, initializeLayers=True):
        self.cost = cost
        if initializeLayers:
            for i in range(0,len(self.layers)):
                if i == len(self.layers) - 1:
                    self.layers[i].initialize(self.getLayerSize(i-1))
                else:
                    self.layers[i].initialize(self.getLayerSize(i-1))

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    # This function returns
    # yp - the output of the network
    # a - a list of inputs for each layer of the newtork where
    #     a[i] is the input to layer i
    #     (note this does not include the network output!)
    # z - a list of values for each layer after evaluating layer.z(a) but
    #     before evaluating the nonlinear function for the layer
    def evaluate(self, x):
        curA = x.T
        a = [curA]
        z = []
        for layer in self.layers:
            z.append(layer.z(curA))
            curA = layer.a(z[-1])
            a.append(curA)
        yp = a.pop()
        return yp, a, z

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    def predict(self, a):
        a,_,_ = self.evaluate(a)
        return a.T

    # Computes the gradients at each layer. y is the true labels, yp is the
    # predicted labels, and z is a list of the intermediate values in each
    # layer. Returns the gradients and the forward pass outputs (per layer).
    #
    # In particular, we compute dMSE/dz_i. The reasoning behind this is that
    # in the update function for the optimizer, we do not give it the z values
    # we compute from evaluating the network.
    def compute_grad(self, x, y):
        # Feed forward, computing outputs of each layer and
        # intermediate outputs before the non-linearities
        yp, a, z = self.evaluate(x)

        # d represents (dMSE / da_i) that you derive in part (e);
        #   it is inialized here to be (dMSE / dyp)
        d = self.cost.dx(y.T, yp)
        grad = []

        # Backpropogate the error
        for layer, curZ in zip(reversed(self.layers),reversed(z)):
            
            ############################
            ########## PART D ##########
            ############################
        
            # Compute the gradient of the output of each layer with respect to the error
            # grad[i] should correspond with the gradient of the output of layer i
            #   before the activation is applied (dMSE / dz_i); be sure values are stored
            #   in the correct ordering!
            grad.insert(0, d * layer.dx(curZ))
            d = layer.W.T @ grad[0]

        return grad, a

    # Computes the gradients at each layer. y is the true labels, yp is the
    # predicted labels, and z is a list of the intermediate values in each
    # layer. Uses numerical derivatives to solve rather than symbolic derivatives.
    # Returns the gradients and the forward pass outputs (per layer).
    #
    # In particular, we compute dMSE/dz_i. The reasoning behind this is that
    # in the update function for the optimizer, we do not give it the z values
    # we compute from evaluating the network.
    def numerical_grad(self, x, y, delta=1e-4):

        # computes the loss function output when starting from the ith layer
        # and inputting z_i
        def compute_cost_from_layer(layer_i, z_i):
            cost = self.layers[layer_i].a(z_i)
            for layer in self.layers[layer_i+1:]:
                cost = layer.a(layer.z(cost))
            return self.cost.fx(y.T, cost)

        # numerically computes the gradient of the error with respect to z_i
        def compute_grad_from_layer(layer_i, inp):
            mask = np.zeros(self.layers[layer_i].b.shape)
            grad_z = []
            # iterate to compute gradient of each variable in z_i, one at a time
            for i in range(mask.shape[0]):
                mask[i] = 1
                delta_p_output = compute_cost_from_layer(layer_i, inp+mask*delta)
                delta_n_output = compute_cost_from_layer(layer_i, inp-mask*delta)
                grad_z.append((delta_p_output - delta_n_output) / (2 * delta))
                mask[i] = 0;

            return np.vstack(grad_z)

        _, a, _ = self.evaluate(x)

        grad = []
        i = 0
        curA = x.T
        for layer in self.layers:
            curA = layer.z(curA)
            grad.append(compute_grad_from_layer(i, curA))
            curA = layer.a(curA)
            i += 1


        return grad, a

    # Train the network given the inputs x and the corresponding observations y
    # The network should be trained for numEpochs iterations using the supplied
    # optimizer
    def train(self, x, y, numEpochs, optimizer):

        # Initialize some stuff
        n = x.shape[0]
        x = x.copy()
        y = y.copy()
        hist = []
        optimizer.initialize(self.layers)

        # Run for the specified number of epochs
        for epoch in range(0,numEpochs):

            # Compute the gradients
            grad, a = self.compute_grad(x, y)

            # Update the network weights
            optimizer.update(self.layers, grad, a)

            # Compute the error at the end of the epoch
            yh = self.predict(x)
            C = self.cost.fx(y, yh)
            C = np.mean(C)
            hist.append(C)
        return hist
    
    ############################
    ########## PART J ##########
    ############################
    
    def trainBatch(self, x, y, batchSize, numEpochs, optimizer):
        hist = []
        for epoch in np.arange(0,numEpochs):
            e = []
            for i in range(0, x.shape[0], batchSize):
                end = min(i+batchSize, x.shape[0])
                batchx = x[i:end, :]
                batchy = y[i:end, :]
                e = e + self.train(batchx, batchy, 1, optimizer)
            hist.append(np.mean(e))
        return hist

if __name__ == '__main__':
    # switch these statements to True to run the code for the corresponding parts
    # PART E
    DEBUG_MODEL = False
    # Part G
    BASE_MODEL = False
    # Part H
    DIFF_SIZES = False
    # Part I
    RIDGE = False
    # Part J
    SGD = False
    # Part K
    PARTk = False



    # Generate the training set
    np.random.seed(9001)
    x=np.random.uniform(-np.pi,np.pi,(1000,1))
    y=np.sin(x)
    xLin=np.linspace(-np.pi,np.pi,250).reshape((-1,1))
    yHats = {}

    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.02,tanh=0.02,linear=0.005)
    names = ['ReLU','linear','tanh']

    #### PART F ####
    if DEBUG_MODEL:
        print('Debugging gradients..')
        # Build the model
        activation = activations["ReLU"]
        model = Model(x.shape[1])
        model.addLayer(DenseLayer(10,activation()))
        model.addLayer(DenseLayer(10,activation()))
        model.addLayer(DenseLayer(1,LinearActivation()))
        model.initialize(QuadraticCost())

        grad, _ = model.compute_grad(x, y)
        n_grad, _ = model.numerical_grad(x, y)
        for i in range(len(grad)):
            print('squared difference of layer %d:' % i, np.linalg.norm(grad[i] - n_grad[i]))


    #### PART G ####
    if BASE_MODEL:
        print('\n----------------------------------------\n')
        print('Standard fully connected network')
        for key in names:
            # Build the model
            activation = activations[key]
            model = Model(x.shape[1])
            model.addLayer(DenseLayer(100,activation()))
            model.addLayer(DenseLayer(100,activation()))
            model.addLayer(DenseLayer(1,LinearActivation()))
            model.initialize(QuadraticCost())

            # Train the model and display the results
            hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
            yHat = model.predict(x)
            yHats[key] = model.predict(xLin)
            error = np.mean(np.square(yHat - y))/2
            print(key+' MSE: '+str(error))
            plt.plot(hist)
            plt.title(key+' Learning curve')
            plt.show()

        # Plot the approximations
        font = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
                'size'   : 12}
        matplotlib.rc('font', **font)
        y = np.sin(xLin)
        for key in activations:
            plt.plot(xLin,y)
            plt.plot(xLin,yHats[key])
            plt.title(key+' approximation')
            plt.savefig(key+'-approx.png')
            plt.show()

    # Train with different sized networks
    #### PART H ####
    if DIFF_SIZES:
        print('\n----------------------------------------\n')
        print('Training with various sized network')
        names = ['ReLU', 'tanh']
        sizes = [5,10,25,50]
        widths = [1,2,3]
        errors = {}
        y = np.sin(x)
        for key in names:
            error = []
            for width in widths:
                for size in sizes:
                    activation = activations[key]
                    model = Model(x.shape[1])
                    for _ in range(width):
                        model.addLayer(DenseLayer(size,activation()))
                    model.addLayer(DenseLayer(1,LinearActivation()))
                    model.initialize(QuadraticCost())
                    hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
                    yHat = model.predict(x)
                    yHats[key] = model.predict(xLin)
                    e = np.mean(np.square(yHat - y))/2
                    error.append(e)
            errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))

        # Print the results
        for key in names:
            error = errors[key]
            print(key+' MSE Error')
            header = '{:^8}'
            for _ in range(len(sizes)):
                header += ' {:^8}'
            headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
            print(header.format(*headerText))
            for width,row in zip(widths,error):
                text = '{:>8}'
                for _ in range(len(row)):
                    text += ' {:<8}'
                rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
                print(text.format(*rowText))

    # Perform ridge regression on the last layer of the network
    #### PART I ####
    if RIDGE:
        print('\n----------------------------------------\n')
        print('Running ridge regression on last layer')
        from sklearn.linear_model import Ridge
        errors = {}
        for key in names:
            error = []
            sizes = [5,10,25,50]
            widths = [1,2,3]
            for width in widths:
                for size in sizes:
                    activation = activations[key]
                    model = Model(x.shape[1])
                    for _ in range(width):
                        model.addLayer(DenseLayer(size,activation()))
                    model.initialize(QuadraticCost())
                    ridge = Ridge(alpha=0.1)
                    X = model.predict(x)
                    ridge.fit(X,y)
                    yHat = ridge.predict(X)
                    e = np.mean(np.square(yHat - y))/2
                    error.append(e)
            errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))

        # Print the results
        for key in names:
            error = errors[key]
            print(key+' MSE Error')
            header = '{:^8}'
            for _ in range(len(sizes)):
                header += ' {:^8}'
            headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
            print(header.format(*headerText))
            for width,row in zip(widths,error):
                text = '{:>8}'
                for _ in range(len(row)):
                    text += ' {:<8}'
                rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
                print(text.format(*rowText))

        # Plot the results
        for key in names:
            for width,row in zip(widths,errors[key]):
                layer = ' layers'
                if width == 1:
                    layer = ' layer'
                plt.semilogy(row,label=str(width)+layer)
            plt.title('MSE for ridge regression with '+key+' activation')
            plt.xticks(range(len(sizes)),sizes)
            plt.xlabel('Layer size')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(key+'-ridge.png')
            plt.show()

    ############################
    ########## PART J ##########
    ############################
    
    if SGD:
        print('\n----------------------------------------\n')
        print('Running SGD')
        batchSizes = [1, 10, 20, 50, 100]
        for key in names:
            for batchSize in batchSizes:
                activation = activation[key]
                model = Model(x.shape[1])
                model.addLayer(DenseLayer(100,activation()))
                model.addLayer(DenseLayer(100,activation()))
                model.addLayer(DenseLayer(1,LinearActivation()))
                model.initialize(QuadraticCost())
                epoch = 25 * batchSize
                hist = model.trainBatch(x, y, batchSize, epoch, GDOptimizer(eta=lr[key]))
                yHat = model.predict(x)
                yHats[key] = model.predict(xLin)
                err = np.mean(np.square(yHat - y))/2
                print(key, 'batchsize:', batchSize, 'MSE:', err)
                
    ############################
    ########## PART K ##########
    ############################
    
    if PARTk:
        print('\n----------------------------------------\n')
        print('Running Bonus Question')
        # we only use ReLU in this question
        activation = activations['ReLU']
        model = Model(x.shape[1])
        model.addLayer(DenseLayer(100,activation()))
        model.addLayer(DenseLayer(100,activation()))
        model.addLayer(DenseLayer(1,LinearActivation()))
        model.initialize(QuadraticCost())

        # Train the model and display the results
        hist = model.train(x,y,100,GDOptimizer(eta=0.01))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('100 max epochs, 0.01 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,100,GDOptimizer(eta=0.05))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('100 max epochs, 0.05 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,100,GDOptimizer(eta=0.1))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('100 max epochs, 0.1 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,200,GDOptimizer(eta=0.01))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('200 max epochs, 0.01 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,200,GDOptimizer(eta=0.05))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('200 max epochs, 0.05 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,200,GDOptimizer(eta=0.1))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('200 max epochs, 0.1 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,500,GDOptimizer(eta=0.01))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('500 max epochs, 0.01 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,500,GDOptimizer(eta=0.05))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('500 max epochs, 0.05 LR'+' MSE: '+str(error))
        
        hist = model.train(x,y,100,GDOptimizer(eta=0.1))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print('500 max epochs, 0.1 LR'+' MSE: '+str(error))







