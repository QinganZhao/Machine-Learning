import numpy as np
import scipy.io
import matplotlib.pyplot as plt

### Problem 6a ###
mdict = scipy.io.loadmat("a.mat")

x = mdict['x']
u = mdict['u']

# Compute a and b
b = x[0][1:]
A = np.vstack((x[0][:-1], u[0][:-1])).T
result = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
print("A:", result[0],"\nB:", result[1])

### Problem 6b ###
mdict = scipy.io.loadmat("b.mat")

x = mdict['x']
u = mdict['u']
x = x.reshape(x.shape[:-1])
u = u.reshape(u.shape[:-1])
X = x[:-1]
U = u[:-1]
b = x[1:]
A = np.hstack((X, U))
result = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b).T
print("A:", result[:,:3], "\n\nB:", result[:,3:])

### Problem 6c ###
mdict = scipy.io.loadmat("train.mat")

# Assemble xu matrix
x = mdict["x"]   # position of a car
v = mdict["xd"]  # velocity of the car
xprev = mdict["xp"]   # position of the car ahead
vprev = mdict["xdp"]  # velocity of the car ahead

acc = mdict["xdd"]  # acceleration of the car

#a, b, c, d, e = 0, 0, 0, 0, 0

# Compute a, b, c, d, e
now = np.vstack([x, v])
prev = np.vstack([xprev, vprev]) 
tmp = np.vstack([now, prev, np.random.random(x.shape[:])]).T
result = acc.dot(tmp).dot(np.linalg.inv(tmp.T.dot(tmp)))
a = result[0][0]
b = result[0][1]
c = result[0][2]
d = result[0][3]
e = result[0][4]

print("Fitted dynamical system:")
print("xdd_i = {:.3f} x_i + {:.3f} xd_i + {:.3f} x_i-1 + {:.3f} xd_i-1 + {:.3f}".format(a, b, c, d, e))

### Problem 7a ###
# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()

# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# Plot three images with label 0 and three images with label 1
j = 0
for i in range(n_train):
    if train_labels[i] == 0:
        visualize_digit(train_features[i,:], train_labels[i])
        j = j + 1
    if j == 3:
        break
j = 0
for i in range(n_train):
    if train_labels[i] == 1:
            visualize_digit(train_features[i,:], train_labels[i])
            j = j + 1
    if j == 3:
        break

### Problem 7b ###
# Linear regression
# Solve the linear regression problem, regressing
X = train_features
y = 2 * train_labels - 1
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
Err = np.linalg.norm(X.dot(W) - y) ** 2

# Report the residual error and the weight vector
print("Residual error:", Err)
print("Weight Vector:\n", W)

### Problem 7c ###
# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = train_labels.shape[0]

# Evaluate the training set
train_result = X.dot(W) 
train_result[train_result <= 0] = int(0)
train_result[train_result > 0] = int(1)
train_acc = (n_train - np.count_nonzero(train_labels - train_result)) / n_train

# Report the training accuracy
print("Training accuracy:", train_acc)

# Evaluate the test set
test_result = test_features.dot(W)
test_result[test_result <= 0] = int(0)
test_result[test_result > 0] = int(1)
test_acc = (n_test - np.count_nonzero(test_labels - test_result)) / n_test

# Report the test accuracy
print("Test accuracy:", test_acc)

### Problem 7e ###
# Use 0 (for class 0) and 1 (for class 1) as the entries
X = train_features
y = train_labels
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
train_result = X.dot(W) 
train_result[train_result <= 0.5] = int(0)
train_result[train_result > 0.5] = int(1)
train_acc = (n_train - np.count_nonzero(train_labels - train_result)) / n_train
print("Training accuracy:", train_acc)
test_result = test_features.dot(W)
test_result[test_result <= 0.5] = int(0)
test_result[test_result > 0.5] = int(1)
test_acc = (n_test - np.count_nonzero(test_labels - test_result)) / n_test
print("Test accuracy:", test_acc)

# Use -1/1 with bias
X = np.hstack((train_features, np.ones((train_features.shape[0],1))))
y = 2 * train_labels - 1
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
train_result = X.dot(W) 
train_result[train_result <= 0] = int(0)
train_result[train_result > 0] = int(1)
train_acc = (n_train - np.count_nonzero(train_labels - train_result)) / n_train
print("Training accuracy:", train_acc)
X_test = np.hstack((test_features, np.ones((test_features.shape[0],1))))
test_result = X_test.dot(W)
test_result[test_result <= 0] = int(0)
test_result[test_result > 0] = int(1)
test_acc = (n_test - np.count_nonzero(test_labels - test_result)) / n_test
print("Test accuracy:", test_acc)


