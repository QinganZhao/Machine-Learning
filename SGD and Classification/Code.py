##### Problem 2 #####
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

X = np.random.normal(scale = 20, size=(100,10))
print(np.linalg.matrix_rank(X)) # confirm that the matrix is full rank
# Theoretical optimal solution
w = np.random.normal(scale = 10, size = (10,1))
y = X.dot(w)

def sgd(X, y, w_actual, threshold, max_iterations, step_size, gd=False):
    if isinstance(step_size, float):
        step_size_func = lambda i: step_size
    else:
        step_size_func = step_size
        
    # run 10 gradient descent at the same time, for averaging purpose
    # w_guesses stands for the current iterates (for each run)
    w_guesses = [np.zeros((X.shape[1], 1)) for _ in range(10)]
    n = X.shape[0]
    error = []
    it = 0
    above_threshold = True
    previous_w = np.array(w_guesses)
    
    while it < max_iterations and above_threshold:
        it += 1
        curr_error = 0
        for j in range(len(w_guesses)):
            if gd:
                # Your code, implement the gradient for GD
                sample_gradient = X.T @ (X @ w_guesses[j] - y)
            else:
                # Your code, implement the gradient for SGD
                tmp = np.random.randint(0, X.shape[0])
                sample_gradient = (X[tmp, :].T * (X[tmp, :] @ w_guesses[j] - y[tmp])).reshape((len(w_guesses), 1))
                
            # Your code: implement the gradient update
            # learning rate at this step is given by step_size_func(it)            
            w_guesses[j] = w_guesses[j] - step_size_func(it) * sample_gradient
            
            curr_error += np.linalg.norm(w_guesses[j]-w_actual)
        error.append(curr_error/10)
        
        diff = np.array(previous_w) - np.array(w_guesses)
        diff = np.mean(np.linalg.norm(diff, axis=1))
        above_threshold = (diff > threshold)
        previous_w = np.array(w_guesses)
    return w_guesses, error

its = 5000
w_guesses, error = sgd(X, y, w, 1e-10, its, 0.0001)

iterations = [i for i in range(len(error))]
#plt.semilogy(iterations, error, label = "Average error in w")
plt.semilogy(iterations, error, label = "Average error in w")
plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=True)
plt.title("Average Error vs Iterations for SGD with exact sol")
plt.legend()
plt.show()

print("Required iterations: ", len(error))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses])
print("Final average error: ", average_error)

y2 = y + np.random.normal(scale=5, size = y.shape)
w=np.linalg.inv(X.T @ X) @ X.T @ y2

its = 5000
w_guesses2, error2 = sgd(X, y2, w, 1e-5, its, 0.0001)
w_guesses3, error3 = sgd(X, y2, w, 1e-5, its, 0.00001)
w_guesses4, error4 = sgd(X, y2, w, 1e-5, its, 0.000001)

w_guess_gd, error_gd = sgd(X, y2, w, 1e-5, its, 0.00001, True)

plt.semilogy([i for i in range(len(error2))], error2, label="SGD, lr = 0.0001")
plt.semilogy([i for i in range(len(error3))], error3, label="SGD, lr = 0.00001")
plt.semilogy([i for i in range(len(error4))], error4, label="SGD, lr = 0.000001")
plt.semilogy([i for i in range(len(error_gd))], error_gd, label="GD, lr = 0.00001")
plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=True)
plt.title("Total Error vs Iterations for SGD without exact sol")
plt.legend()
plt.show()

print("Required iterations, lr = 0.0001: ", len(error2))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses2])
print("Final average error: ", average_error)

print("Required iterations, lr = 0.00001: ", len(error3))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses3])
print("Final average error: ", average_error)

print("Required iterations, lr = 0.000001: ", len(error4))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses4])
print("Final average error: ", average_error)

print("Required iterations, GD: ", len(error_gd))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guess_gd])
print("Final average error: ", average_error)

its = 5000
def step_size(step):
    if step < 500:
        return 1e-4 
    if step < 1500:
        return 1e-5
    if step < 3000:
        return 3e-6
    return 1e-6

w_guesses_variable, error_variable = sgd(X, y2, w, 1e-10, its, step_size, False)

plt.semilogy([i for i in range(len(error_variable))], error_variable, label="Average error, decreasing lr")
plt.semilogy([i for i in range(len(error2))], error2, label="Average error, lr = 0.0001")
plt.semilogy([i for i in range(len(error3))], error3, label="Average error, lr = 0.00001")
plt.semilogy([i for i in range(len(error4))], error4, label="Average error, lr = 0.000001")

plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=True)
plt.title("Error vs Iterations for SGD with no exact sol")
plt.legend()
plt.show()

print("Required iterations, variable lr: ", len(error_variable))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses_variable])
print("Average error with decreasing lr:", average_error)

##### Problem 3 #####

import time

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def optimize(x, y, pred, loss, optimizer, training_epochs, batch_size):
    acc = []
    with tf.Session() as sess:  # start training
        sess.run(tf.global_variables_initializer())  # Run the initializer
        for epoch in range(training_epochs):  # Training cycle
            avg_loss = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_loss += c / total_batch

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = accuracy_.eval({x: mnist.test.images, y: mnist.test.labels})
            acc.append(accuracy)
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss),
                  "accuracy={:.9f}".format(accuracy))
    return acc


def train_linear(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean((y - pred)**2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def train_logistic(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.matmul(x, W) + b
    loss = tf.losses.softmax_cross_entropy(y, pred)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def train_nn(learning_rate=0.01, training_epochs=50, batch_size=50, n_hidden=64):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([784, n_hidden]))
    W2 = tf.Variable(tf.random_normal([n_hidden, 10]))
    b1 = tf.Variable(tf.random_normal([n_hidden]))
    b2 = tf.Variable(tf.random_normal([10]))

    pred = tf.matmul(tf.tanh(tf.matmul(x, W1) + b1), W2) + b2
    loss = tf.losses.softmax_cross_entropy(y, pred)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def main():
    for batch_size in [50, 100, 200]:
        time_start = time.time()
        acc_linear = train_linear(batch_size=batch_size)
        print("train_linear finishes in %.3fs" % (time.time() - time_start))

        plt.plot(acc_linear, label="linear bs=%d" % batch_size)
        plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.show()
    
    acc_logistic = train_logistic()
    plt.plot(acc_logistic, label="logistic regression")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.show()
    
    acc_nn = train_nn()
    plt.plot(acc_nn, label="neural network")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.show()

if __name__ == "__main__":
    tf.set_random_seed(0)
    main()

### Part e ###

import numpy as np

import tensorflow as tf

n_data = 6000
n_dim = 50

w_true = np.random.uniform(low=-2.0, high=2.0, size=[n_dim])

x_true = np.random.uniform(low=-10.0, high=10.0, size=[n_data, n_dim])
x_ob = x_true + np.random.randn(n_data, n_dim)
y_ob = x_true @ w_true + np.random.randn(n_data)

learning_rate = 0.01
training_epochs = 100
batch_size = 100


def main():
    x = tf.placeholder(tf.float32, [None, n_dim])
    y = tf.placeholder(tf.float32, [None, 1])

    w = tf.Variable(tf.random_normal([n_dim, 1]))

    tmp = y - tf.matmul(x, w)
    cost = n_data / 2 * tf.log(tf.norm(w) ** 2 + 1) + tf.matmul(tf.transpose(tmp), tmp)

    # Adam is a fancier version of SGD, which is insensitive to the learning
    # rate.  Try replace this with GradientDescentOptimizer and tune the
    # parameters!
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w_sgd = sess.run(w).flatten()

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_data / batch_size)
            for i in range(total_batch):
                start, end = i * batch_size, (i + 1) * batch_size
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={
                        x: x_ob[start:end, :],
                        y: y_ob[start:end, np.newaxis]
                    })
                avg_cost += c / total_batch
            w_sgd = sess.run(w).flatten()
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(float(avg_cost)), 
                  "|w-w_true|^2 = {:.9f}".format(np.sum((w_sgd - w_true)**2)))
            
    # Total least squares: SVD
    X = x_true
    y = y_ob
    stacked_mat = np.hstack((X, y[:, np.newaxis])).astype(np.float32)
    u, s, vh = np.linalg.svd(stacked_mat)
    w_tls = -vh[-1, :-1] / vh[-1, -1]

    error = np.sum(np.square(w_tls - w_true))
    print("\nTLS through SVD error: |w-w_true|^2 = {}".format(error))


if __name__ == "__main__":
    tf.set_random_seed(0)
    np.random.seed(0)
    main()