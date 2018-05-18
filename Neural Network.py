# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:33:46 2018

@author: Vineet Joshi
"""
import numpy as np
import scipy.io
from math import sqrt

# Function to calculate sigmoid
def sigmoid(z):
	"""" Returns the sigmoid of element z, irrespective of whether z is a scalar, vector or matrix
	"""
	ans = 1/(1+np.exp(-z))
	return ans

# Function to calculate gradient of the sigmoid
def sigmoidgradient(z):
	""" Returns the gradient of the sigmoid of z.
	Works for scalars, vectors, and matrices too.
	"""
	ans = np.multiply(sigmoid(z), (1 - sigmoid(z)))
	return ans


# Function to initialize weights randomly
def randinitialWeights(L_in, L_out):
	""" Randomly initializes weights for the parameters in THETA for the number 
	of nodes in the input layer and the output layer
	The function outputs a matrix of shape L_out X (L_in +1) 
	The +1 is to include the bias unit whose value is always 1. """
	epsilon = sqrt(6)/(sqrt(L_in + L_out))
	W = np.random.rand(L_out, L_in + 1)*2*epsilon - epsilon
	return W		 


# Importing the MNIST dataset
mat = scipy.io.loadmat('mnist.mat')

# Importing our training data
data_X = mat['trainX']
data_y = mat['trainY'].T

# Normalizing the data
data_X = data_X/255.0

# Next, we will split our training data into training and validation sets
# We will use the first 50000 examples for our training set and the remainder
# of 10000 examples as our validation set.

X_train = data_X[:50000, : ]

X_val = data_X[50000:, : ]

y_train = data_y[:50000, : ]

y_val = data_y[50000:, : ]

# Importing the test set
X_test = mat['testX']
y_test = mat['testY'].T

# Normalizing the test set
X_test = X_test/255.0


# For the purposes of training our model, we need to change the y labels into 
# vectors of boolean values.

# So, for each label between 0 and 9, we will have a vector of length 10 where 
# the ith element will be 1 if the label equals i

# To convert y values into boolean vectors for training set 
train_labels = np.empty((len(y_train), 10))

for i in range(len(y_train)):
	x = np.arange(10)
	labels = np.array([x == y_train[i]]).astype(int)
	train_labels[i,:] = labels


# To convert y values into boolean vectors for validation set 
val_labels = np.empty((len(y_val), 10))

for i in range(len(y_val)):
	x = np.arange(10)
	labels = np.array([x == y_val[i]]).astype(int)
	val_labels[i,:] = labels
	

# To convert y values into boolean vectors for test set 
test_labels = np.empty((len(y_test), 10))

for i in range(len(y_test)):
	x = np.arange(10)
	labels = np.array([x == y_test[i]]).astype(int)
	test_labels[i,:] = labels

# Setting up parameters for our neural network. These do not include the bias unit.
input_layer_size = 784
hidden_layer_size = 40
output_layer = 10

# Setting our regularization parameter
Lambda = 0.05

# Get the number of training examples
m = np.size(X_train, axis = 0)


# Initialize values for the weights going from the input layer to the hidden layer
# and the values for the weights going from the hidden layer to the output layer
init_Theta1 = randinitialWeights(input_layer_size, hidden_layer_size)
init_Theta2 = randinitialWeights(hidden_layer_size, output_layer)

# We will flatten the weight matrices to create a column vector containing all the weights
weights = np.concatenate((init_Theta1.flatten(), init_Theta2.flatten()), 0)

# FORWARD PROPOGATION
def forwardpropogation(X, param1, param2):
	""" Implements the forward propagation of our neural network using the
	sigmoidal activation function
	
	X is our data
	param1 are the weights from our input layer to the hidden layer
	param2 are the weights from our hidden layer to the output layer
	
	The function outputs z2 and z3 which are the activations of the 
	hidden layer and the activations of the output layer respectively.
	"""
	
	# Adding a column of ones which we will be our bias unit
	X = np.hstack((np.ones((len(X), 1)), X))
	
	a2 = np.dot(X, param1.T)
	z2 = sigmoid(a2)

	z2 = np.hstack((np.ones((len(z2), 1)), z2))
	
	a3 = np.dot(z2, param2.T)
	z3 = sigmoid(a3)
	
	return (z2, z3)


# BACKPROPAGATION
def backpropagation(weights, *args):	
	# COST FUNCTION WITH REGULARIZATION 
	""" Calculates the regularized cost function of the neural network and 
	implements the backpropagation algorithm to compute the gradient of the
	cost function.
	weights is a vector containing the values of our initial parameters
	Additional arguments of the function are 
	X - pixel data of each image 
	y - labels in the form of boolean vectors
	Lambda - Our regularization parameter
	
	The function outputs the regularized cost funciton and its gradient as an
	array
	
	"""
	X, y, Lambda = args
	
	# Obtain our initial weights using the weights array
	Theta1 = np.reshape(weights[:np.size(init_Theta1)], newshape = np.shape(init_Theta1))
	Theta2 = np.reshape(weights[np.size(init_Theta1):], newshape = np.shape(init_Theta2))

	# Run forward propagation to calculate the activations of the hidden layer
	# and the output layer
	z2, z3 = forwardpropogation(X = X_train, param1=Theta1, param2=Theta2)
	
	# Add a column of ones to our data(the bias unit)
	X = np.hstack((np.ones((len(X), 1)), X))
	
	# Define and calculate our cost function
	# We will use the log loss cost function with regularization
	J = np.add(np.multiply(y, np.log(z3)), np.multiply((1 - y), np.log(1 - z3)))
	J = -np.sum(J)/len(y)
	
	# Define our regularization term
	# While regularization, we must take care to ensure that the weights 
	# of the bias term are excluded
	reg_term = np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:]))
	
	# Calculate cost function with regularization
	J = J + Lambda*reg_term/(2*m)
	
	# Errors in the output layer
	err = z3 - y
	
	# Error in the hidden layer
	d2 = np.multiply(np.dot(err, Theta2[:, 1:]), sigmoidgradient(np.dot(X, Theta1.T)))
	
	# Calcualte the gradients for Theta1 and Theta 2
	Theta1_grad = np.dot(d2.T, X)/m
	Theta2_grad = np.dot(err.T, z2)/m
		     
	# Regularizing the parameters except the first columns which contain 
	# weights for the bias unit
	Theta1[:,1:] = (Lambda/m)*Theta1[:,1:]
	Theta2[:,1:] = (Lambda/m)*Theta2[:,1:]
	
	# We get the gradients using regularization
	Theta1_grad = np.column_stack((Theta1_grad[:,0], np.add(Theta1_grad[:,1:], Theta1[:, 1:])))
	Theta2_grad = np.column_stack((Theta2_grad[:,0], np.add(Theta2_grad[:,1:], Theta2[:, 1:])))
	
	# To finish, we will flatten our gradient into an array
	grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()), 0)
	
	return(J, grad)



# Set our arguments for the backpropagation algorithm
args = (X_train, train_labels, Lambda)

# Optimize our weights using the Conjugate Gradient algorithm
final_weights = scipy.optimize.minimize(backpropagation, weights, method = 'CG', args = args,
				       jac=True, options = {'maxiter':50})


# Obtain our Theta matrics from the optimized weights
T1 = np.reshape(final_weights.x[:np.size(init_Theta1)], newshape = np.shape(init_Theta1))
T2 = np.reshape(final_weights.x[np.size(init_Theta1):], newshape = np.shape(init_Theta2))


# PREDICTION
def prediction(X, y, param1, param2):
	""" Returns the predictions of our neural network along with the 
	accuracy of the model
	X - images data
	y - image labels
	param1 - optimized weights from the input layer to the hidden layer
	param2 - optimized weights from the hidden layer to the output layer
	"""
	l2, pred = forwardpropogation(X , param1, param2)
	o = np.empty((len(X), 1))
	for i in range(len(X)):
		o[i] = np.argmax(pred[i])

	accu = np.mean(o == y)

	return (o, accu*100)	
	
# Time to run our model and make predictions on our training, validation, and
# test sets.
o_train, accu_train = prediction(X_train, y_train, T1, T2)
o_val, accu_val = prediction(X_val, y_val, T1, T2)
o_test, accu_test = prediction(X_test, y_test, T1, T2)

print('The accuracy on our training set is', accu_train, '%')
print('The accuracy on our validation set is', accu_val, '%')
print('The accuracy on our test set is', accu_test, '%')


