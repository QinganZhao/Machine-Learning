from common import *
########################################################################
######### Part b ###################################
########################################################################

########################################################################
#########  Gradient Computing and MLE ###################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc, sensor_loc, 
								single_distance):
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
	grad = np.zeros_like(single_obj_loc)
	#Your code: implement the gradient of loglikelihood

	return grad

def find_mle_by_grad_descent_part_b(initial_obj_loc, 
		   sensor_loc, single_distance, lr=0.001, num_iters = 10000):
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
	# Your code: do gradient descent
		
	return obj_loc
if __name__ == "__main__":	
	########################################################################
	#########  MAIN ########################################################
	########################################################################

	# Your code: set some appropriate learning rate here
	lr = 1.0

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