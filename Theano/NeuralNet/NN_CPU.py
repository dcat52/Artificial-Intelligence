import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import theano
import theano.tensor as T
import math
#import pydot
#from IPython.display import Image
#from IPython.display import SVG
import timeit

# Display plots inline and change default figure size
#%matplotlib inline
#matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
# train_X, train_y = sklearn.datasets.make_moons(200, noise=0.20)
# train_X = train_X.astype(np.float32)
# train_y = train_y.astype(np.int32)
# plt.scatter(train_X[:,0], train_X[:,1], s=40, c=train_y, cmap=plt.cm.Spectral)

class NN():
	def __init__(
		self, 
		train_X, train_y, 
		input_dim=2, hidden_dim=5, output_dim=2, 
		num_passes=5000, 
		block=False, UID=None, buff=0.5, h=0.01
		):

		# Size definitions
		self.num_examples = len(train_X) # training set size
		self.nn_input_dim = input_dim # input layer dimensionality
		self.nn_output_dim = output_dim # output layer dimensionality
		self.nn_hdim = hidden_dim # hiden layer dimensionality

		# Gradient descent parameters (I picked these by hand)
		self.epsilon = 0.01 # learning rate for gradient descent
		self.reg_lambda = 0.01 # regularization strength

		self.h = h
		self.buff = buff

		self.train_X = train_X
		self.train_y = train_y

		self.num_passes = num_passes

		# Our data vectors
		self.X = T.matrix('X') # matrix of doubles
		self.y = T.lvector('y') # vector of int64

		# Shared variables with initial values. We need to learn these.
		self.W1 = theano.shared(np.random.randn(self.nn_input_dim, self.nn_hdim), name='W1')
		self.b1 = theano.shared(np.zeros(self.nn_hdim), name='b1')
		self.W2 = theano.shared(np.random.randn(self.nn_hdim, self.nn_output_dim), name='W2')
		self.b2 = theano.shared(np.zeros(self.nn_output_dim), name='b2')

		# Forward propagation
		# Note: We are just defining the expressions, nothing is evaluated here!
		self.z1 = self.X.dot(self.W1) + self.b1
		self.a1 = T.tanh(self.z1)
		self.z2 = self.a1.dot(self.W2) + self.b2
		self.y_hat = T.nnet.softmax(self.z2) # output probabilties

		# The regularization term (optional)
		self.loss_reg = 1./self.num_examples * self.reg_lambda/2 * (T.sum(T.sqr(self.W1)) + T.sum(T.sqr(self.W2))) 
		# the loss function we want to optimize
		self.loss = T.nnet.categorical_crossentropy(self.y_hat, self.y).mean() + self.loss_reg

		# Returns a class prediction
		self.prediction = T.argmax(self.y_hat, axis=1)

		# Theano functions that can be called from our Python code
		self.forward_prop = theano.function([self.X], self.y_hat)
		self.calculate_loss = theano.function([self.X, self.y], self.loss)
		self.predict = theano.function([self.X], self.prediction)

		# Easy: Let Theano calculate the derivatives for us!
		self.dW2 = T.grad(self.loss, self.W2)
		self.db2 = T.grad(self.loss, self.b2)
		self.dW1 = T.grad(self.loss, self.W1)
		self.db1 = T.grad(self.loss, self.b1)

		self.gradient_step = theano.function(
			[self.X, self.y],
			updates=((self.W2, self.W2 - self.epsilon * self.dW2),
					 (self.W1, self.W1 - self.epsilon * self.dW1),
					 (self.b2, self.b2 - self.epsilon * self.db2),
					 (self.b1, self.b1 - self.epsilon * self.db1)))

		# Build a model with a 3-dimensional hidden layer
		self.build_model(print_loss=True)

		# Plot the decision boundary
		plt.title("%d classes - %d hidden layer width - %d iterations" % (self.nn_output_dim, self.nn_hdim, self.num_passes))
		self.plot_decision_boundary(lambda x: self.predict(x))
		plt.show(block=False)

	# Helper function to plot a decision boundary.
	# If you don't fully understand this function don't worry, it just generates the contour plot.
	def plot_decision_boundary(self, pred_func):
		# Set min and max values and give it some padding
		x_min, x_max = self.train_X[:, 0].min() - self.buff, self.train_X[:, 0].max() + self.buff
		y_min, y_max = self.train_X[:, 1].min() - self.buff, self.train_X[:, 1].max() + self.buff

		# Generate a grid of points with distance h between them
		xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h), np.arange(y_min, y_max, self.h))
		# Predict the function value for the whole gid
		Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		# Plot the contour and training examples
		plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
		plt.scatter(self.train_X[:, 0], self.train_X[:, 1], c=self.train_y, cmap=plt.cm.Spectral)


	# This function learns parameters for the neural network and returns the model.
	# - num_passes: Number of passes through the training data for gradient descent
	# - print_loss: If True, print the loss every 1000 iterations
	def build_model(self, num_passes=100000, print_loss=False):
		
		# Re-Initialize the parameters to random values. We need to learn these.
		# (Needed in case we call this function multiple times)
		np.random.seed(0)
		self.W1.set_value(np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim))
		self.b1.set_value(np.zeros(self.nn_hdim))
		self.W2.set_value(np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim))
		self.b2.set_value(np.zeros(self.nn_output_dim))
		
		# Gradient descent. For each batch...
		for i in xrange(1, self.num_passes+1):
			# This will update our parameters W2, b2, W1 and b1!
			self.gradient_step(self.train_X, self.train_y)
			
			# if i % 500 == 0:
			# Optionally print the loss.
			# This is expensive because it uses the whole dataset, so we don't want to do it too often.
			if print_loss and (math.log(i)/math.log(2) % 1 == 0 and i <= 1024) or (i > 1024 and i <= 10240 and i%1024==0) or (i > 10240 and i <= 102400 and i%10240==0) or (i > 102400 and i%51200==0):
				#print "Loss after iteration %i: %f" %(i, self.calculate_loss(self.train_X, self.train_y))
				self.update_graph(i)

	def update_graph(self,i):
		plt.title("%d classes - %d hidden layer width - %d iterations" % (self.nn_output_dim, self.nn_hdim, i))
		self.plot_decision_boundary(lambda x: self.predict(x))
		plt.show(block=False)
		plt.pause(0.03)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import uniform
import time

# np.random.seed(0)

# random.seed(0)
# A,B,C = [],[],[]

# num = 40

# for i in range(0,num):
# 	A.append([uniform(0,3),uniform(0,2.5)])
# 	B.append([uniform(2.0,6),uniform(1.5,3)])
# 	C.append([uniform(5,7),uniform(0,2.5)])

# A = 2*np.asarray(A)
# B = 2*np.asarray(B)
# C = 2*np.asarray(C)

# train_X = np.concatenate((A,B))
# train_X = np.concatenate((train_X,C))
# train_y = num*[0]
# train_y.extend(num*[1])
# train_y.extend(num*[2])
# train_y = np.asarray(train_y)

np.random.seed(0)
A = (5*np.random.random_sample((50,2))) + (2*np.random.random_sample((1,2))-1)
B = (5*np.random.random_sample((50,2))) + (2*np.random.random_sample((1,2))-1) +  3
train_X = np.concatenate((A,B))
train_y = 50*[0]
train_y.extend(50*[1])
plt.figure(0)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.Spectral)
plt.show()

a = NN(train_X, train_y, num_passes=1048576, input_dim=2, output_dim=2, hidden_dim=10)
print "Done!"
plt.show()
