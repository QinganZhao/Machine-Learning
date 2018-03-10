from common import *
from part_b_starter import find_mle_by_grad_descent_part_b
from part_b_starter import compute_gradient_of_likelihood
from part_c_starter import log_likelihood

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
	# Your code: finish the grad loglike
		
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
	# Your code: finish the gradient descent

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
	estimated_distance = scipy.spatial.distance.cdist(obj_loc, 
											sensor_loc, 
											metric='euclidean')
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
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    # Your code: compute the mse for this case                  
              
print('The MSE for Case 1 is {}'.format(mse))

########################################################################
#########  Case 2. #####################################################
########################################################################
mse =0
        
for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = False)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    # Your code: compute the mse for this case

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

print('The MSE for Case 2 (if we knew mu is [300,300]) is {}'.format(mse)) 
