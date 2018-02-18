# imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
np.random.seed(0)

def generate_data(n):
    """
    This function generates data of size n.
    """
    #TODO implement this
    return (X,y)

def tikhonov_regression(X,Y,Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    #TODO implement this
    return w

def compute_mean_var(X,y,Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    #TODO implement this
    return mux,muy,sigmax,sigmay,sigmaxy

# Define the sigmas and number of samples to use
Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
Num_Sample_Range = [5, 500]

##############################################################

def gen_plot():
    """
    This function refreshes the interactive plot.
    """
    plt.sca(ax)
    plt.cla()
    CS = plt.contour(X_grid, Y_grid, Z, levels =
        np.concatenate([np.arange(0,0.05,0.01),np.arange(0.05,1,0.05)]))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sigma'+ names[i] + ' with num_data = {}'.format(num_data))

names = [str(i) for i in range(1,len(Sigmas)+1)]

fig = plt.figure(figsize=(7.5,7.5))
ax = plt.axes()
plt.subplots_adjust(left=0.15, bottom=0.3)

# define the interactive sliders
sigma_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
sample_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
sigma_slider = Slider(sigma_ax, 'Sigma', valmin=0, valmax=len(Sigmas)-1e-5,
                         valinit=0, valfmt="%d")
num_data_slider = Slider(sample_ax, 'Num Samples', valmin=Num_Sample_Range[0],
                         valmax=Num_Sample_Range[1], valinit=Num_Sample_Range[0],
                         valfmt="%d")
sigma_slider.valtext.set_visible(False)
num_data_slider.valtext.set_visible(False)

# initial settings for plot
num_data = Num_Sample_Range[0]; Sigma = Sigmas[0]; i = 0
x = np.arange(0.5, 1.5, 0.01)
y = np.arange(0.5, 1.5, 0.01)
X_grid, Y_grid = np.meshgrid(x, y)
# Generate the function values of bivariate normal.
X, Y = generate_data(num_data)
mux,muy,sigmax,sigmay,sigmaxy = compute_mean_var(X,Y,Sigma)
Z = matplotlib.mlab.bivariate_normal(X_grid,Y_grid, sigmax, sigmay, mux, muy, sigmaxy)

def sigma_update(val):
    """
    This function is called in response to interaction with the Sigma sliding bar.
    """
    global Z, i
    if val != -1:
        i = int(val)
    Sigma = Sigmas[i]
    mux,muy,sigmax,sigmay,sigmaxy = compute_mean_var(X,Y,Sigma)
    Z = matplotlib.mlab.bivariate_normal(X_grid,Y_grid, sigmax, sigmay, mux, muy, sigmaxy)
    gen_plot()

def num_sample_update(val):
    """
    This function is called in response to interaction with the number of samples sliding bar.
    """
    global X, Y, num_data
    max_val = Num_Sample_Range[1]
    min_val = Num_Sample_Range[0]
    r = max_val - min_val
    num_data_ = int(((val - min_val) / r)**2 * r + min_val)
    if num_data == num_data_:
        return
    num_data = num_data_
    X, Y = generate_data(num_data)
    sigma_update(-1)

sigma_slider.on_changed(sigma_update)
num_data_slider.on_changed(num_sample_update)

gen_plot()
plt.show()
