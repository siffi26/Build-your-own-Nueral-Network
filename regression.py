import csv
from numpy import genfromtxt
import numpy as np
import math
import matplotlib.pyplot as plt
#--------------------------------------------------All functions that will be needed!-------------------------------------------------
def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	cache = Z
	return A, cache
def sigmoid_backward(dA, cache):
	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)
	assert (dZ.shape == Z.shape)
	return dZ
def relu(Z):
	A = np.maximum(0,Z)
	assert(A.shape == Z.shape)
	cache = Z 
	return A, cache
def relu_backward(dA, cache):
	Z = cache
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.
	dZ[Z <= 0] = 0
	assert (dZ.shape == Z.shape)
	return dZ
def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	return Z, cache
def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis=1, keepdims=True) / m
	dA_prev = np.dot(W.T, dZ)
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	return dA_prev, dW, db
#----------------------------------------------Initialization--------------------------------------------------------
def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters = {}
	L = len(layer_dims)
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])#*0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	return parameters
#------------------------------------------Forward Propogation--------------------------------------------------------
def linear_activation_forward(A_prev, W, b, activation):
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	elif activation == "linear":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A = Z
		activation_cache = (A_prev, W, b)
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)
	return A, cache
def L_model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters) // 2  # number of layers in the neural network
	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "linear")
	caches.append(cache)
	assert(AL.shape == (1,X.shape[1]))
	return AL, caches
#-----------------------------------------------Backward Propogation--------------------------------------------------------
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "linear":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache) 
    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	dAL = (AL-Y)*(2/m)
	current_cache = caches[L-1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "linear")
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
	return grads
#------------------------------------------------Stochastic Gradient descent--------------------------------------------------------
def random_mini_batches(X, Y, mini_batch_size = 1, seed = 0): #mini_batch_size=1 for stocahstic gradient descent
	np.random.seed(seed)            
	m = X.shape[1]                  # number of training examples
	mini_batches = []
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))
	num_complete_minibatches = math.floor(m/mini_batch_size) 
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches
#------------------------------------------------Update Parameters--------------------------------------------------------
def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2 # number of layers in the neural network
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
	return parameters
#-------------------------------------After model is trained using predict-------------------------------------------------
def predict(X, y, parameters):
	m = X.shape[1]
	n = len(parameters) // 2 # number of layers in the neural network
	p = np.zeros((1,m))
	# Forward propagation
	probas, caches = L_model_forward(X, parameters)
	# convert probas to 0/1 predictions
	AL = probas
	Y = y
	x1 = np.arange(1, m+1)
	x1 = np.reshape(x1,(1, m)).T
	y1 = Y.T
	y2 = probas.T
	plt.plot(x1, y1, linestyle='-', color='b')
	plt.plot(x1, y2, linestyle='-', color='r')
	plt.show()
	print("RMS error: " + str(compute_cost(AL, Y, "RMS")))
	#print("Accuracy: "  + str(np.sum((p == y)/m)))
	#return p
#------------------------------------------------Loss function--------------------------------------------------------------
def compute_cost(AL, Y, ctype):
	m = Y.shape[1]
	if ctype=="LSE":
		cost = np.sum((AL-Y)**2)
	elif ctype=="RMS":
		cost = np.sqrt((1/m)*np.sum((AL-Y)**2))
	#cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m #cross-entropy cost
	cost = np.squeeze(cost) 
	assert(cost.shape == ())
	return cost
#-------------------------------------------------Nueral network Model(Combining All)----------------------------------------
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0095, num_iterations = 3000, print_cost=True):
	np.random.seed(1)
	costs = []
	m = X[0].size
	print("Here's m: ", m)
	parameters = initialize_parameters_deep(layers_dims)
	for i in range(0, num_iterations):
		#for j in range(0, m):
			#minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
		# Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
		AL, caches = L_model_forward(X, parameters)
		# Compute cost.
		cost = compute_cost(AL, Y, "LSE")
		# Backward propagation.
		grads = L_model_backward(AL, Y, caches)
		# Update parameters.
		parameters = update_parameters(parameters, grads, learning_rate)
		# Print the cost every 100 training example
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	return parameters
#-----------------------------------------------------one hot---------------------------------------------------
def onehot(value, L):
	m = value.min()
	mask = np.concatenate((np.array([[1]]), np.zeros((1, L-1)) ), axis=1)
	lis=[np.roll(mask,i) for i in range(L)]
	vector = np.zeros((1, L))
	for i in value:
		vector = np.concatenate((vector, lis[i-m]),axis=0)
	return vector[1:,:]
#-----------------------------------------------------main---------------------------------------------------
layers_dims = [15, 14, 10, 1] #  4-layer model
my_data = np.array(list(csv.reader(open('energy_efficiency_data.csv', 'r'), delimiter=',')))
key = my_data[0,0:8][np.newaxis,:]
Y = (my_data[1:,8])[np.newaxis,:].astype(float)
X = (my_data[1:,0:8]).astype(float)
orient = X[:,5]  
glaze= X[:,7]
X = np.delete(X,[5,7],1)
X = X/np.amax(X, 0)
vector_orient = onehot(orient.astype(int), 4)
vector_glaze = onehot(glaze.astype(int), 6)[:,1:]
X = np.concatenate((X, vector_orient, vector_glaze), axis=1)
YT = np.transpose(Y)
X = np.concatenate((X, YT), axis =1)
#np.random.shuffle(X)
train_x = X[:576,:15].astype(float).T
train_y = X[:576,15].astype(float).T
train_y = np.reshape(train_y,(1,576))
test_x = X[576:,:15].astype(float).T
test_y = X[576:,15].astype(float).T
test_y = np.reshape(test_y,(1,192))
#print(test_x.shape)
#print(test_y.shape)
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True)
predict(train_x, train_y, parameters)
predict(test_x, test_y, parameters)