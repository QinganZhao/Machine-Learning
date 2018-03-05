### Problem 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread,imsave

imFile = 'stpeters_probe_small.png'
compositeFile = 'tennis.png'
targetFile = 'interior.jpg'

# This loads and returns all of the images needed for the problem
# data - the image of the spherical mirror
# tennis - the image of the tennis ball that we will relight
# target - the image that we will paste the tennis ball onto
def loadImages():
    imFile = 'stpeters_probe_small.png'
    compositeFile = 'tennis.png'
    targetFile = 'interior.jpg'
    
    data = imread(imFile).astype('float')*1.5
    tennis = imread(compositeFile).astype('float')
    target = imread(targetFile).astype('float')/255 

    return data, tennis, target
    

# This function takes as input a square image of size m x m x c
# where c is the number of color channels in the image.  We
# assume that the image contains a scphere and that the edges
# of the sphere touch the edge of the image.
# The output is a tuple (ns, vs) where ns is an n x 3 matrix
# where each row is a unit vector of the direction of incoming light
# vs is an n x c vector where the ith row corresponds with the
# image intensity of incoming light from the corresponding row in ns
def extractNormals(img):

    # Assumes the image is square
    d = img.shape[0]
    r = d / 2
    ns = []
    vs = []
    for i in range(d):
        for j in range(d):

            # Determine if the pixel is on the sphere
            x = j - r
            y = i - r
            if x*x + y*y > r*r-100:
                continue

            # Figure out the normal vector at the point
            # We assume that the image is an orthographic projection
            z = np.sqrt(r*r-x*x-y*y)
            n = np.asarray([x,y,z])
            n = n / np.sqrt(np.sum(np.square(n)))
            view = np.asarray([0,0,-1])
            n = 2*n*(np.sum(n*view))-view
            ns.append(n)
            vs.append(img[i,j])

    return np.asarray(ns), np.asarray(vs)

# This function renders a diffuse sphere of radius r
# using the spherical harmonic coefficients given in
# the input coeff where coeff is a 9 x c matrix
# with c being the number of color channels
# The output is an 2r x 2r x c image of a diffuse sphere
# and the value of -1 on the image where there is no sphere
def renderSphere(r,coeff):

    d = 2*r
    img = -np.ones((d,d,3))
    ns = []
    ps = []

    for i in range(d):
        for j in range(d):

            # Determine if the pixel is on the sphere
            x = j - r
            y = i - r
            if x*x + y*y > r*r:
                continue

            # Figure out the normal vector at the point
            # We assume that the image is an orthographic projection
            z = np.sqrt(r*r-x*x-y*y)
            n = np.asarray([x,y,z])
            n = n / np.sqrt(np.sum(np.square(n)))
            ns.append(n)
            ps.append((i,j))

    ns = np.asarray(ns) 
    
    # scale the output
    ns = ns / 384
    
    B = computeBasis(ns)
    vs = B.dot(coeff) 

    for p,v in zip(ps,vs):
        img[p[0],p[1]] = np.clip(v,0,255)

    return img

# relights the sphere in img, which is assumed to be a square image
# coeff is the matrix of spherical harmonic coefficients
def relightSphere(img, coeff):
    img = renderSphere(int(img.shape[0]/2),coeff)/255*img/255 
    return img

# Copies the image of source onto target
# pixels with values of -1 in source will not be copied
def compositeImages(source, target):
    
    # Assumes that all pixels not equal to 0 should be copied
    out = target.copy()
    cx = int(target.shape[1]/2)
    cy = int(target.shape[0]/2)
    sx = cx - int(source.shape[1]/2)
    sy = cy - int(source.shape[0]/2)

    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            if np.sum(source[i,j]) >= 0:
                out[sy+i,sx+j] = source[i,j]

    return out

# Fill in this function to compute the basis functions
# This function is used in renderSphere()
def computeBasis(ns):
    # Returns the first 9 spherical harmonic basis functions

    #################################################
    # TODO: Compute the first 9 basis functions
    #################################################
    B = np.zeros((len(ns), 9))
    for i in range(len(ns)):
        B[i, 0] = 1
        B[i, 1] = ns[i, 1]
        B[i, 2] = ns[i, 0]
        B[i, 3] = ns[i, 2]
        B[i, 4] = ns[i, 0] * ns[i, 1]
        B[i, 5] = ns[i, 1] * ns[i, 2]
        B[i, 6] = 3 * ns[i, 2] ** 2 - 1
        B[i, 7] = ns[i, 0] * ns[i, 2]
        B[i, 8] = ns[i, 0] ** 2 - ns[i, 1] ** 2
    
    return B

if __name__ == '__main__':

    data,tennis,target = loadImages()
    ns, vs = extractNormals(data)    
    B = computeBasis(ns)

    # reduce the number of samples because computing the SVD on
    # the entire data set takes too long
    Bp = B[::50]
    vsp = vs[::50]
    
    #################################################
    # TODO: Solve for the coefficients using least squares
    # or total least squares here
    ##################################################
    
    # OLS
    #coeff = np.linalg.inv(Bp.T @ Bp) @ Bp.T @ vsp
    
    # TLS
    BV = np.hstack([Bp, vsp])
    VT = np.linalg.svd(BV)[2]
    V = VT.T
    Vxy = V[0:Bp.shape[1],Bp.shape[1]:]     
    Vyy = V[Bp.shape[1]:,Bp.shape[1]:] 
    Vyy_inv = np.linalg.inv(VYY) 
    coeff = -Vxy @ Vyy_inv 
    
    img = relightSphere(tennis,coeff) 

    output = compositeImages(img,target) 

    print('Coefficients:\n'+str(coeff))

    plt.figure(1)
    plt.imshow(output)
    plt.show()

    #imsave('output.png',output)

### Problem 3 

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 80
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 300}
plt.rc('font', **font)
sns.set()


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
    '''
    d = original dimension
    k = projected dimension
    '''
    return 1./np.sqrt(k)*np.random.normal(0, 1, (d, k))

def random_proj(X, k):
    _, d= X.shape
    return X.dot(random_matrix(d, k))

## PCA and projections ##
def my_pca(X, k):
    '''
    compute PCA components
    X = data matrix (each row as a sample)
    k = #principal components
    '''
    n, d = X.shape
    assert(d>=k)
    _, _, Vh = np.linalg.svd(X)    
    V = Vh.T
    return V[:, :k]

def pca_proj(X, k):
    '''
    compute projection of matrix X
    along its first k principal components
    '''
    P = my_pca(X, k)
    # P = P.dot(P.T)
    return X.dot(P)


######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from random projection of X, versus y
    for binary classification for y in {-1, 1}
    '''
    
    # test train split
    _, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # random projection
    J = np.random.normal(0., 1., (d, k))
    rand_proj_X = X_train.dot(J)
    
    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_proj_X, y_train)
    
    # predict y
    y_pred=line.predict(X_test.dot(J))
    
    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)

def pca_proj_accuracy(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from PCA projection of X, versus y
    for binary classification for y in {-1, 1}
    '''

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca projection
    P = my_pca(X_train, k)
    P = P.dot(P.T)
    pca_proj_X = X_train.dot(P)
                
    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_proj_X, y_train)
    
     # predict y
    y_pred=line.predict(X_test.dot(P))
    

    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)

######## LOADING THE DATASETS #########

# to load the data:
data1 = np.load('data1.npz')
X1 = data['X']
y1 = data['y']
n1, d1 = X1.shape

data1 = np.load('data2.npz')
X2 = data['X']
y2 = data['y']
n2, d2 = X2.shape

data3 = np.load('data3.npz')
X3 = data['X']
y3 = data['y']
n3, d3 = X3.shape

n_trials = 10  # to average for accuracies over random projections

######### YOUR CODE GOES HERE ##########

# Using PCA and Random Projection for:
# Visualizing the datasets
## Part h
plt.figure()
plt.scatter(random_proj(X1, 2)[:, 0], random_proj(X1, 2)[:, 1], label='Random projects')
plt.scatter(pca_proj(X1, 2)[:, 0], pca_proj(X1, 2)[:, 1], label='PCA')
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

plt.figure()
plt.scatter(random_proj(X2, 2)[:, 0], random_proj(X2, 2)[:, 1], label='Random projects')
plt.scatter(pca_proj(X2, 2)[:, 0], pca_proj(X2, 2)[:, 1], label='PCA')
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

plt.figure()
plt.scatter(random_proj(X3, 2)[:, 0], random_proj(X3, 2)[:, 1], label='Random projects')
plt.scatter(pca_proj(X3, 2)[:, 0], pca_proj(X3, 2)[:, 1], label='PCA')
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

# Computing the accuracies over different datasets.
## Part i
# dataset 1
pcaAcc1 = np.zeros(d1)
randAcc1 = np.zeros(d1)
for i in range(d1):
    pcaAcc1[i] = pca_proj_accuracy(X1, y1, i+1) 
    randAcc1[i] = rand_proj_accuracy_split(X1, y1, i+1)
plt.figure()
plt.plot(np.arange(1,d1+1), randAcc1, label='Random projects')
plt.plot(np.arange(1,d1+1), pcaAcc1, label='PCA')
plt.xlabel('k', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

# dataset 2
pcaAcc2 = np.zeros(d2)
randAcc2 = np.zeros(d2)
for i in range(d2):
    pcaAcc2[i] = pca_proj_accuracy(X2, y2, i+1) 
    randAcc2[i] = rand_proj_accuracy_split(X2, y2, i+1)
plt.figure()
plt.plot(np.arange(1,d1+1), randAcc1, label='Random projects')
plt.plot(np.arange(1,d1+1), pcaAcc1, label='PCA')
plt.xlabel('k', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

# dataset 3
pcaAcc3 = np.zeros(d3)
randAcc3 = np.zeros(d3)
for i in range(d3):
    pcaAcc1[i] = pca_proj_accuracy(X3, y3, i+1) 
    randAcc1[i] = rand_proj_accuracy_split(X3, y3, i+1)
plt.figure()
plt.plot(np.arange(1,d3+1), randAcc1, label='Random projects')
plt.plot(np.arange(1,d3+1), pcaAcc1, label='PCA')
plt.xlabel('k', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

# And computing the SVD of the feature matrix
## Part j
sing1 = np.linalg.svd(X1, compute_uv=False)
sing2 = np.linalg.svd(X2, compute_uv=False)
sing3 = np.linalg.svd(X3, compute_uv=False)
plt.figure()
plt.plot(np.arange(1,len(sing1)+1), sing1, label='Dataset 1')
plt.plot(np.arange(1,len(sing2)+1), sing2, label='Dataset 2')
plt.plot(np.arange(1,len(sing3)+1), sing3, label='Dataset 3')
plt.xlabel('Singular values', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.show()

