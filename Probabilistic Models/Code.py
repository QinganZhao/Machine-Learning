### Problem 2e

import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125,625]
plt.figure(figsize=[12, 10])             
for k in range(4):
    n = sample_size[k]
    
    # generate data
    X = np.random.normal(0, 1, n)
    Z = np.random.uniform(-0.5, -0.5, n)
    N = 1001
    Wtrue = 1 # set true w0 = 1
    Y = Wtrue * X
    W = np.linspace(0, 2, N) # compute likelihood within [0,2]
    # np.linspace, np.random.normal and np.random.uniform might be useful functions

    
    likelihood = np.ones(N) # likelihood as a function of w

    for i1 in range(N):
        # compute likelihood
        w = W[i1]
        for i2 in range(n):
            if abs(Y[i2] - w * X[i2]) > 0.5:
                likelihood[i1] = 0

    likelihood /= sum(likelihood) # normalize the likelihood
    
    plt.figure()
    # plotting likelihood for different n
    plt.plot(W, likelihood)
    plt.xlabel('w', fontsize=10)
    plt.title(['n=' + str(n)], fontsize=14)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125]
sigma = [1, 2, 3] # set three different sigmas: 1, 2, 3
W1s_true = [5.01, 3.42, 0.45] # set true W based on part h
W2s_true = [5.25, -1.55, 0.39]
# simulate X and Y; let X be normally distributed, Y be uniformly distributed
for j in range(3):
    print('sigma =', sigma[j], ':\n')
    W1_true = W1s_true[j]
    W2_true = W2s_true[j]
    for k in range(3):
        n = sample_size[k]
        print('sample size =', n, ':\n')
        # generate data 
        # np.linspace, np.random.normal and np.random.uniform might be useful functions
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)
        Y = W1_true * X1 + W2_true * X2 + Z
        # compute likelihood
        N = 101 
        W1s = np.linspace(-10, 10, N)
        W2s = np.linspace(-10, 10, N)
        likelihood = np.ones([N,N]) # likelihood as a function of w_1 and w_0

        for i1 in range(N):
            w_1 = W1s[i1]
            for i2 in range(N):
                w_2 = W2s[i2]
                L = 1
                for i in range(n):
                    L = L * np.exp(-0.5 * (Y[i] - X1[i] * w_1 - X2[i] * w_2) ** 2)
                L = L * np.exp(-0.5 * (w_1 ** 2 + w_2 ** 2) / sigma[j])
                # compute the likelihood
                likelihood[i1][i2] = L

        # plotting the likelihood
        plt.figure()                          
        # for 2D likelihood using imshow
        plt.imshow(likelihood, cmap='hot', aspect='auto',extent=[-10,10,-10,10])
        plt.xlabel('w0')
        plt.ylabel('w1')
        plt.show()

### Problem 4e
import numpy as np
import matplotlib.pyplot as plt

# assign problem parameters
D = 20
n = 50
err = np.zeros([D, n])

# generate data
# np.random might be useful
X = np.random.uniform(-1, 1, 2*n)
Z = np.random.normal(0, 1, 2*n)
Ystar = X + 1
Y = X + 1 + Z

# fit data with different models
# np.polyfit and np.polyval might be useful
for i in range(0, n):
    N = i + 2*n
    for d in range(1, D+1):
        fit = np.polyfit(X, Y, d)
        tmp = 0
        for j in range(N):
            tmp = tmp + (np.polyval(fit, X[j]) - Ystar[j]) ** 2
        err[d-1][i] = tmp/N
    X = np.append(X, np.random.uniform(-1, 1, 1)) 
    Z = np.append(Z, np.random.normal(0, 1, 1))
    Ystar = X + 1
    Y = X + 1 + Z
    
# plotting figures
# sample code

plt.figure()
plt.subplot(121)
plt.semilogy(np.arange(1, D+1),err[:,-1])
plt.xlabel('degree of polynomial')
plt.ylabel('log of error')
plt.subplot(122)
plt.semilogy(np.arange(2*n, 2*n+n), err[-1,:])
plt.xlabel('number of samples')
plt.ylabel('log of error')
plt.show()


### Problem 4g
import numpy as np
import matplotlib.pyplot as plt

# assign problem parameters
D = 20
n = int(120 / 3)
err = np.zeros([D, n])

# generate data
# np.random might be useful
X = np.random.uniform(-1, 1, 2*n)
Z = np.random.normal(0, 1, 2*n)
Ystar = np.exp(X)
Y = np.exp(X) + Z

# fit data with different models
# np.polyfit and np.polyval might be useful
for i in range(0, n):
    N = i + 2*n
    for d in range(1, D+1):
        fit = np.polyfit(X, Y, d)
        tmp = 0
        for j in range(N):
            tmp = tmp + (np.polyval(fit, X[j]) - Ystar[j]) ** 2
        err[d-1][i] = tmp/N
    X = np.append(X, np.random.uniform(-1, 1, 1)) 
    Z = np.append(Z, np.random.normal(0, 1, 1))
    Ystar = X + 1
    Y = X + 1 + Z
    
# plotting figures
# sample code

plt.figure()
plt.subplot(121)
plt.semilogy(np.arange(1, D+1),err[:,-1])
plt.xlabel('degree of polynomial')
plt.ylabel('log of error')
plt.subplot(122)
plt.semilogy(np.arange(2*n, 2*n+n), err[-1,:])
plt.xlabel('number of samples')
plt.ylabel('log of error')
plt.show()


### Problem 5a
import pickle
import matplotlib.pyplot as plt
import numpy as np

x_train = pickle.load(open('x_train.p','rb'), encoding='latin1')
y_train = pickle.load(open('y_train.p','rb'), encoding='latin1')
x_test = pickle.load(open('x_test.p','rb'), encoding='latin1')
y_test = pickle.load(open('y_test.p','rb'), encoding='latin1')

def visualize(pic, i):
    plt.imshow(pic)
    plt.xlabel(str(i) + 'th image')
    plt.show()
    print("Corresponding control vector:", y_train[i, :])

visualize(x_train[0,:], 0)
visualize(x_train[10,:], 10)
visualize(x_train[20,:], 20)
    
### Problem 5b
x_train_new = x_train.reshape(91, 2700)
y_train_new = y_train.reshape(91, 3)
W = np.linalg.inv(x_train_new.T.dot(x_train_new)).dot(x_train_new.T).dot(y_train_new) #Singluar matrix

### Problem 5c
lam = [0.1, 1, 10, 100, 1000]
def ridge(training_set, Lambda):
    return np.linalg.inv(training_set.T.dot(training_set) + Lambda * np.identity(2700)).dot(training_set.T).dot(y_train_new)
W = [ridge(x_train_new, lam[0]), ridge(x_train_new, lam[1]), ridge(x_train_new, lam[2]), ridge(x_train_new, lam[3]), ridge(x_train_new, lam[4])]
err = np.zeros(5)

for i in range(5):
    Sum = 0
    for j in range(91):
        Sum = Sum + np.linalg.norm(x_train_new[j].T @ W[i] - y_train_new[j])
    err[i] = Sum / 91 
    print('For lambda =', lam[i], ', error:', err[i])
    

### Problem 5d
x_train_std = x_train_new /255 * 2 + 1
W_std = [ridge(x_train_std, lam[0]), ridge(x_train_std, lam[1]), ridge(x_train_std, lam[2]), ridge(x_train_std, lam[3]), ridge(x_train_std, lam[4])]
err = np.zeros(5)
for i in range(5):
    Sum = 0
    for j in range(91):
        Sum = Sum + np.linalg.norm(x_train_new[j].T @ W_std[i] - y_train_new[j])
    err[i] = Sum / 91
    print('For lambda =', lam[i], ', error:', err[i])

### Problem 5e
x_test_new = x_test.reshape(62, 2700)
y_test_new = y_test.reshape(62, 3)

print("Without standardization:")
for i in range(5):
    Sum = 0
    for j in range(62):
        Sum = Sum + np.linalg.norm(x_test_new[j].T @ W[i] - y_test_new[j])
    err[i] = Sum / 62 
    print('For lambda =', lam[i], ', error:', err[i])

print("\nWith standardization:")
for i in range(5):
    Sum = 0
    for j in range(62):
        Sum = Sum + np.linalg.norm(x_test_new[j].T @ W_std[i] - y_test_new[j])
    err[i] = Sum / 62
    print('For lambda =', lam[i], ', error:', err[i])




