### Problem 4b
import numpy as np
import matplotlib.pyplot as plt
import os
import math

plot_col = ['r', 'g', 'b', 'k', 'm']
plot_mark = ['o', '^', 'v', 'D', 'x', '+']

# Plots the rows in 'ymat' on the y-axis vs. 'xvec' on the x-axis
# with labels 'ylabels'
# and saves figure as pdf to 'dirname/filename' 
def plotmatnsave(ymat, xvec, ylabels, dirname, filename):
    no_lines = len(ymat)
    fig = plt.figure(0)

    if len(ylabels) > 1:
        for i in range(no_lines):
            xs = np.array(xvec)
            ys = np.array(ymat[i])
            plt.plot(xs, ys, color = plot_col[i % len(plot_col)], lw=1, label=ylabels[i])
        
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    savepath = os.path.join(dirname, filename)
    plt.xlabel('$x$', labelpad=10)
    plt.ylabel('$f(x)$', labelpad=10)
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

# Sets the labels
labels = ['$e^x$', '1st order', '2nd order', '3rd order', '4th order']

# TODO: Given x values in "x_vec", save the respective function values e^x,
# and its first to fourth degree Taylor approximations
# as rows in the matrix "y_mat"
x_vec = np.arange(-4, 3.1, 0.1)
y_mat = [np.exp(x_vec), 1+x_vec, 1+x_vec+0.5*x_vec**2, 1+x_vec+0.5*x_vec**2+x_vec**2/6, 1+x_vec+0.5*x_vec**2+x_vec**2/6+x_vec**4/24]


# Define filename, invoke plotmatnsave
filename = 'approx_plot.pdf'
plotmatnsave(y_mat, x_vec, labels, '.', filename)

# Plot the second figure
x_vec2 = np.arange(-20, 8.1, 0.1)
y_mat2 = [np.exp(x_vec), 1+x_vec, 1+x_vec+0.5*x_vec**2, 1+x_vec+0.5*x_vec**2+x_vec**2/6, 1+x_vec+0.5*x_vec**2+x_vec**2/6+x_vec**4/24]
filename2 = 'approx_plot2.pdf'
plotmatnsave(y_mat2, x_vec2, labels, '.', filename2)

### Problem 5b

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T

    n = 20  # max degree
    err = np.zeros(n - 1)

    # fill in err
    A = np.array([1] * n)
    for i in range(n-1):
        D = i + 1
        for j in range(D+1):
            if j==0:
                A = np.array([1] * n)
            else:
                A = np.vstack([x_train ** j, A])
        A = A.T
        W = lstsq(A, y_train) 
        y_hat = A @ W
        err[i] = (np.linalg.norm(y_train - y_hat) ** 2)/n
    # YOUR CODE HERE

    plt.plot(err)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Training Error')
    plt.show()


if __name__ == "__main__":
    main()

# problem 6d

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    y_fresh = np.array(data['y_fresh']).T

    n = 20  # max degree
    err_train = np.zeros(n - 1)
    err_fresh = np.zeros(n - 1)

    # fill in err_fresh and err_train
    for i in range(n-1):
        D = i + 1
        for j in range(D+1):
            if j==0:
                A = np.array([1] * n)
            else:
                A = np.vstack([x_train ** j, A])
        A = A.T
        W = lstsq(A, y_train)
        y_hat = A @ W
        err_train[i] = (np.linalg.norm(y_train - y_hat) ** 2)/n
        err_fresh[i] = (np.linalg.norm(y_fresh - y_hat) ** 2)/n
    # YOUR CODE HERE

    plt.figure()
    plt.ylim([0, 6])
    plt.plot(err_train, label='train')
    plt.plot(err_fresh, label='fresh')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

### Problem 6b
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
W = np.linalg.inv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(ytrain)

Xtest = np.load("Xtest.npy")
ytest = np.load("ytest.npy")

# TODO: report the classification accuracy on the test set
test_result = Xtest.dot(W)
test_result[test_result <= 0] = int(0)
test_result[test_result > 0] = int(1)
test_acc = (len(ytest) - np.count_nonzero(ytest - test_result)) / len(ytest)
print("Test accuracy:", test_acc)

# TODO: Create a matrix Phi_train with polynomial features from the training data
# and solve the linear regression on the training data
X_poly_train = np.hstack([[Xtrain[:,0], Xtrain[:,1], Xtrain[:,0] ** 2, Xtrain[:,0] * Xtrain[:,1], Xtrain[:,1] ** 2, np.ones(len(ytest))]])
W = np.linalg.inv(X_poly_train.T.dot(X_poly_train)).dot(X_poly_train.T).dot(ytrain)
# TODO: Create a matrix Phi_test with polynomial features from the test data
# and report the classification accuracy on the test set








