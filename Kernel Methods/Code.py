### Problem 2c
import numpy as np
import matplotlib.pyplot as plt
mu = [15, 5]

plt.figure(num=1, figsize=(20, 20), facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
sigma1 = [[20, 0], [0, 10]]
samples1 = np.random.multivariate_normal(mu, sigma1, size=100)
plt.scatter(samples1[:, 0], samples1[:, 1])
sample1_mean = np.array([np.mean(samples1[:,0]), np.mean(samples1[:,1])])
sample1_var = sum([np.array([samples1[i] - sample1_mean]).T @ np.array([samples1[i] - sample1_mean]) for i in range(100)]) / 100
xlabel1 = 'Mean:\n'+ str(sample1_mean) + '\nCovariance:\n' + str(sample1_var)
plt.xlabel(xlabel1, fontsize=13)

plt.subplot(2,2,2)
sigma2 = [[20, 14], [14, 10]]
samples2 = np.random.multivariate_normal(mu, sigma, size=100)
plt.scatter(samples2[:, 0], samples2[:, 1])
sample2_mean = np.array([np.mean(samples2[:,0]), np.mean(samples2[:,1])])
sample2_var = sum([np.array([samples2[i] - sample2_mean]).T @ np.array([samples2[i] - sample2_mean]) for i in range(100)]) / 100
xlabel2 = 'Mean:\n'+ str(sample2_mean) + '\nCovariance:\n' + str(sample2_var)
plt.xlabel(xlabel2, fontsize=13)

plt.subplot(2,2,3)
sigma3 = [[20, -14], [-14, 10]]
samples3 = np.random.multivariate_normal(mu, sigma, size=100)
plt.scatter(samples3[:, 0], samples3[:, 1])
sample3_mean = np.array([np.mean(samples3[:,0]), np.mean(samples3[:,1])])
sample3_var = sum([np.array([samples3[i] - sample3_mean]).T @ np.array([samples3[i] - sample3_mean]) for i in range(100)]) / 100
xlabel3 = 'Mean:\n'+ str(sample3_mean) + '\nCovariance:\n' + str(sample3_var)
plt.xlabel(xlabel3, fontsize=13)
plt.show()

### Problem 3d
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def generate_data(n):
    """
    This function generates data of size n.
    """
    #TODO implement this
    X = np.random.randn(n, 2) * np.sqrt(5)
    Z = np.random.randn(n)
    y = np.sum(X, axis=1) + Z
    return (X,y)

def tikhonov_regression(X,Y,Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    #TODO implement this
    w = np.linalg.inv((X.T @ X + np.linalg.inv(Sigma))) @ X.T @ Y
    return w

def compute_mean_var(X,y,Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    #TODO implement this
    mux, muy = tikhonov_regression(X, y, Sigma)
    var = np.linalg.inv(X.T @ X + np.linalg.inv(Sigma))
    sigmax = np.sqrt(var[0, 0])
    sigmay = np.sqrt(var[1, 1])
    sigmaxy = var[0, 1]
    return mux,muy,sigmax,sigmay,sigmaxy

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = [str(i) for i in range(1,6+1)]

for num_data in [5,50,500]:
    X,Y = generate_data(num_data)
    plt.figure(figsize=(30,20))
    for i,Sigma in enumerate(Sigmas):

        mux,muy,sigmax,sigmay,sigmaxy = compute_mean_var(X,Y,Sigma)# TODO compute the mean and covariance of posterior.

        x = np.arange(0.5, 1.5, 0.01)
        y = np.arange(0.5, 1.5, 0.01)
        X_grid, Y_grid = np.meshgrid(x, y)

        Z = matplotlib.mlab.bivariate_normal(X_grid, Y_grid, sigmax, sigmay, mux, muy, sigmaxy)# TODO Generate the function values of bivariate normal.

        # plot
        plt.subplot(2,3,i+1)
        CS = plt.contour(X_grid, Y_grid, Z,
                         levels = np.concatenate([np.arange(0,0.05,0.01),np.arange(0.05,1,0.05)]))
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sigma'+ names[i] + ' with num_data = {}'.format(num_data))
        #plt.savefig('Sigma'+ names[i] + '_num_data_{}.png'.format(num_data))
    plt.show()

### Problem 3e
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
w = [1.0,1.0]
n_test = 100
n_trains = np.arange(5,205,5)
n_trails = 500

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = ['Sigma{}'.format(i+1) for i in range(6)]

def generate_data(n):
    """
    This function generates data of size n.
    """
    #TODO implement this
    X = np.random.randn(n, 2) * np.sqrt(5)
    Z = np.random.randn(n)
    y = np.sum(X, axis=1) + Z
    return (X,y)

def tikhonov_regression(X,Y,Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    #TODO implement this
    w = np.linalg.inv((X.T @ X + np.linalg.inv(Sigma))) @ X.T @ Y
    return w

def compute_mse(X,Y, w):
    """
    This function computes MSE given data and estimated w.
    """
    #TODO implement this
    mse = np.mean((X @ w - Y) ** 2)
    return mse

def compute_theoretical_mse(w):
    """
    This function computes theoretical MSE given estimated w.
    """
    #TODO implement this
    theoretical_mse = 5 * (w[0] - 1) ** 2 + 5 * (w[1] - 1) ** 2 + 1
    return theoretical_mse

# Generate Test Data.
X_test, y_test = generate_data(n_test)

mses = np.zeros((len(Sigmas), len(n_trains), n_trails))

theoretical_mses = np.zeros((len(Sigmas), len(n_trains), n_trails))

for seed in range(n_trails):
    np.random.seed(seed)
    for i,Sigma in enumerate(Sigmas):
        for j,n_train in enumerate(n_trains):
            X_train, y_train = generate_data(n_train)
            w = tikhonov_regression(X_train, y_train, Sigma)
            mses[i, j, seed] = compute_mse(X_test, y_test, w)
            theoretical_mses[i, j, seed] = compute_theoretical_mse(w)

# Plot
plt.figure()
for i,_ in enumerate(Sigmas):
    plt.plot(n_trains, np.mean(mses[i],axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
#plt.savefig('MSE.png')

plt.figure()
for i,_ in enumerate(Sigmas):
    plt.plot(n_trains, np.mean(theoretical_mses[i],axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
#plt.savefig('theoretical_MSE.png')


plt.figure()
for i,_ in enumerate(Sigmas):
    plt.loglog(n_trains, np.mean(theoretical_mses[i]-1,axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
#plt.savefig('log_theoretical_MSE.png')
plt.show()

### Problem 5a

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# choose the data you want to load
data = np.load('circle.npz')
# data = np.load('heart.npz')
# data = np.load('asymmetric.npz')

SPLIT = 0.8
X = data["x"]
y = data["y"]
X /= np.max(X)  # normalize the data

n_train = int(X.shape[0] * SPLIT)
X_train = X[:n_train:, :]
X_valid = X[n_train:, :]
y_train = y[:n_train]
y_valid = y[n_train:]

LAMBDA = 0.001


def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)


def heatmap(f, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx = yy = np.linspace(np.min(X), np.max(X), 72)
    x0, y0 = np.meshgrid(xx, yy)
    x0, y0 = x0.ravel(), y0.ravel()
    z0 = f(x0, y0)

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.hexbin(x0, y0, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx, yy, z0.reshape(xx.size, yy.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    #plt.show()

plt.figure(figsize=(30,8))

plt.subplot(1, 3, 1)
data = np.load('circle.npz')
X = data["x"]
y = data["y"]
X /= np.max(X)
heatmap(lambda x, y: x * x + y * y)

plt.subplot(1, 3, 2)
data = np.load('heart.npz')
X = data["x"]
y = data["y"]
X /= np.max(X)
heatmap(lambda x, y: x * x + y * y)

plt.subplot(1, 3, 3)
data = np.load('asymmetric.npz')
X = data["x"]
y = data["y"]
X /= np.max(X)
heatmap(lambda x, y: x * x + y * y)

plt.show()

### Problem 5b
X = data['x']
Y = data['y']
train_x = X[0:int(X.shape[0]*0.8), :]
train_y = y[0:int(len(y) * 0.8), :]
val_x = X[int(X.shape[0]*0.8):, :]
val_y = y[int(len(y) * 0.8):, :]
p_max = 16 # max D = 6

feat_train = 0

def fit(D)
    w = lstsq(feat_train, train_y, 0.001)
    train_err = np.mean((train_y - feat_train @ w)**2)
    val_err = np.mean((valid_y - feat_val @ w)**2)

# Reference: HW2 solution
def assemble_feature(x, p):
    n_feature = x.shape[1]
    Q = [(np.ones(x.shape[0]), 0, 0)]
    i = 0
    while Q[i][1] < D:
        cx, degree, last_index = Q[i]
        for j in range(last_index, n_feature):
            Q.append((cx * x[:, j], degree + 1, j))
            i += 1
    return np.column_stack([q[0] for q in Q])

for p in range(p_max):
    global feat_x
    feat_train = assemble_feature(train_x, p + 1)
    feat_val = assemble_feature(val_x, p + 1)

