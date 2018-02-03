import numpy as np
import matplotlib.pyplot as plt

Xtrain = np.load("Xtrain.npy")
ytrain = np.load("ytrain.npy")

def visualize_dataset(X, y):
    plt.scatter(X[y < 0.0, 0], X[y < 0.0, 1])
    plt.scatter(X[y > 0.0, 0], X[y > 0.0, 1])
    plt.show()

# visualize the dataset:
visualize_dataset(Xtrain, ytrain)

# TODO: solve the linear regression on the training data

Xtest = np.load("Xtest.npy")
ytest = np.load("ytest.npy")

# TODO: report the classification accuracy on the test set

# TODO: Create a matrix Phi_train with polynomial features from the training data
# and solve the linear regression on the training data

# TODO: Create a matrix Phi_test with polynomial features from the test data
# and report the classification accuracy on the test set
