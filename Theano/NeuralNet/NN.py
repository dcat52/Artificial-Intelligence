import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import theano.tensor as T
import theano

# Neural Net 2D data
# Graphs data too

class NN():
	def __init__(self, train_X, train_y, input_dim=2, hidden_dim=5, output_dim=2, num_passes=5000, block=False, UID=None, buff=0.5, h=0.1):
		if UID is None:
			UID = self
		self.UID = UID

		self.buff = buff
		self.h = h

		self.train_X = train_X.astype(np.float32)
		self.train_y = train_y.astype(np.int32)
		self.train_y_onehot = np.eye(output_dim)[train_y]

		self.X = theano.shared(self.train_X.astype('float32'))
		self.y = theano.shared(self.train_y_onehot.astype('float32'))

		self.num_examples = len(self.train_X)
		self.nn_input_dim = input_dim
		self.nn_hdim = hidden_dim
		self.nn_output_dim = output_dim
		self.num_passes = num_passes

		self.epsilon = np.float32(0.01)
		self.reg_lambda = np.float32(0.01)

		self.W1 = theano.shared(np.random.randn(self.nn_input_dim, self.nn_hdim).astype('float32'), name='W1')
		self.b1 = theano.shared(np.zeros(self.nn_hdim).astype('float32'), name='b1')
		self.W2 = theano.shared(np.random.randn(self.nn_hdim, self.nn_output_dim).astype('float32'), name='W2')
		self.b2 = theano.shared(np.zeros(self.nn_hdim).astype('float32'), name='b2')

		self.z1 = self.X.dot(self.W1) + self.b1
		self.a1 = T.tanh(self.z1)
		self.z2 = self.a1.dot(self.W2) + self.b2
		self.y_hat = T.nnet.softmax(self.z2)

		self.loss_reg = 1./self.num_examples * self.reg_lambda/2 * (T.sum(T.sqr(self.W1)) + T.sum(T.sqr(self.W2)))
		self.loss = T.nnet.categorical_crossentropy(self.y_hat, self.train_y).mean() + self.loss_reg

		self.prediction = T.argmax(self.y_hat, axis=1)

		self.dW2 = T.grad(self.loss, self.W2)
		self.db2 = T.grad(self.loss, self.b2)
		self.dW1 = T.grad(self.loss, self.W1)
		self.db1 = T.grad(self.loss, self.b1)

		self.forward_prop = theano.function([], self.y_hat)
		self.calculate_loss = theano.function([], self.loss)
		self.predict = theano.function([], self.prediction)

		self.gradient_step = theano.function([], updates=(
			(self.W2, self.W2 - self.epsilon * self.dW2),
			(self.W1, self. W1 - self.epsilon * self.dW1),
			(self.b2, self.b2 - self.epsilon * self.db2),
			(self.b1, self.b1 - self.epsilon * self.db1)))

		#plt.figure(UID)
		self.build_model(self.num_passes, print_loss=True)
		self.plot_decision_boundary(self.predict)
		self.title = "%d classes - %d hidden layer width - %d iterations" % (self.nn_output_dim, self.nn_hdim, self.num_passes)
		#plt.title("%d classes - %d hidden layer width - %d iterations" % (self.nn_output_dim, self.nn_hdim, self.num_passes))
		#plt.show(block=block)
		#plt.pause(0.1)

	def build_model(self, num_passes=20000, print_loss=True):
		np.random.seed(0)
		self.W1.set_value((np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim)).astype('float32'))
		self.b1.set_value(np.zeros(self.nn_hdim).astype('float32'))
		self.W2.set_value((np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim)).astype('float32'))
		self.b2.set_value(np.zeros(self.nn_output_dim).astype('float32'))

		for i in xrange(0, num_passes):

			self.gradient_step()

			if print_loss and i%1000==0:
				print "Loss after iteration %i: %f" %(i, self.calculate_loss())

	def plot_decision_boundary(self, pred_func):
		x_min, x_max = self.train_X[:,0].min() - self.buff, self.train_X[:, 0].max() + self.buff
		y_min, y_max = self.train_X[:,1].min() - self.buff, self.train_X[:, 1].max() + self.buff

		xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h), np.arange(y_min, y_max, self.h))

		self.X.set_value((np.c_[xx.ravel(), yy.ravel()]).astype('float32'))

		self.Z = pred_func()
		self.Z = self.Z.reshape(xx.shape)

		self.xx = xx
		self.yy = yy
		# plt.contourf(xx, yy, self.Z, cmap=plt.cm.Spectral)
		# plt.scatter(self.train_X[:,0], self.train_X[:,1], c=self.train_y, s=20, cmap=plt.cm.Spectral)



# Neural Network 3D data
# **WIP**
# Adjusts weights appropriately BUT graphing fails

class NN3D():
	def __init__(self, train_X, train_y, input_dim=2, hidden_dim=5, output_dim=2, num_passes=5000, block=False, UID=None, buff=0.5, h=0.1):
		if UID is None:
			UID = self
		self.UID = UID

		self.buff = buff
		self.h = h

		self.train_X = train_X.astype(np.float32)
		self.train_y = train_y.astype(np.int32)
		self.train_y_onehot = np.eye(output_dim)[train_y]

		self.X = theano.shared(self.train_X.astype('float32'))
		self.y = theano.shared(self.train_y_onehot.astype('float32'))

		self.num_examples = len(self.train_X)
		self.nn_input_dim = input_dim
		self.nn_hdim = hidden_dim
		self.nn_output_dim = output_dim
		self.num_passes = num_passes

		self.epsilon = np.float32(0.01)
		self.reg_lambda = np.float32(0.01)

		self.W1 = theano.shared(np.random.randn(self.nn_input_dim, self.nn_hdim).astype('float32'), name='W1')
		self.b1 = theano.shared(np.zeros(self.nn_hdim).astype('float32'), name='b1')
		self.W2 = theano.shared(np.random.randn(self.nn_hdim, self.nn_output_dim).astype('float32'), name='W2')
		self.b2 = theano.shared(np.zeros(self.nn_hdim).astype('float32'), name='b2')

		self.z1 = self.X.dot(self.W1) + self.b1
		self.a1 = T.tanh(self.z1)
		self.z2 = self.a1.dot(self.W2) + self.b2
		self.y_hat = T.nnet.softmax(self.z2)

		self.loss_reg = 1./self.num_examples * self.reg_lambda/2 * (T.sum(T.sqr(self.W1)) + T.sum(T.sqr(self.W2)))
		self.loss = T.nnet.categorical_crossentropy(self.y_hat, self.train_y).mean() + self.loss_reg

		self.prediction = T.argmax(self.y_hat, axis=1)

		self.dW2 = T.grad(self.loss, self.W2)
		self.db2 = T.grad(self.loss, self.b2)
		self.dW1 = T.grad(self.loss, self.W1)
		self.db1 = T.grad(self.loss, self.b1)

		self.forward_prop = theano.function([], self.y_hat)
		self.calculate_loss = theano.function([], self.loss)
		self.predict = theano.function([], self.prediction)

		self.gradient_step = theano.function([], updates=(
			(self.W2, self.W2 - self.epsilon * self.dW2),
			(self.W1, self. W1 - self.epsilon * self.dW1),
			(self.b2, self.b2 - self.epsilon * self.db2),
			(self.b1, self.b1 - self.epsilon * self.db1)))

		self.build_model(self.num_passes, print_loss=True)

		self.fig = plt.figure(UID)
		self.ax = self.fig.add_subplot(111,projection='3d')

		self.plot_decision_boundary(self.predict)
		plt.title("%d classes - %d hidden layer width - %d iterations" % (self.nn_output_dim, self.nn_hdim, self.num_passes))
		plt.show(block=block)
		plt.pause(0.1)

	def build_model(self, num_passes=20000, print_loss=False):
		np.random.seed(0)
		self.W1.set_value((np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim)).astype('float32'))
		self.b1.set_value(np.zeros(self.nn_hdim).astype('float32'))
		self.W2.set_value((np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim)).astype('float32'))
		self.b2.set_value(np.zeros(self.nn_output_dim).astype('float32'))

		for i in xrange(0, num_passes):

			self.gradient_step()

			if print_loss and i%1000==0:
				print "Loss after iteration %i: %f" %(i, self.calculate_loss())

	def plot_decision_boundary(self, pred_func):
		x_min, x_max = self.train_X[:,0].min() - self.buff, self.train_X[:, 0].max() + self.buff
		y_min, y_max = self.train_X[:,1].min() - self.buff, self.train_X[:, 1].max() + self.buff
		z_min, z_max = self.train_X[:,2].min() - self.buff, self.train_X[:, 2].max() + self.buff

		xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, self.h), np.arange(y_min, y_max, self.h), np.arange(z_min, z_max, self.h))
		self.X.set_value((np.c_[xx.ravel(), yy.ravel(), zz.ravel()]).astype('float32'))
		

		self.Z = pred_func()

		print self.Z
		self.Z = self.Z.reshape(xx.shape)

		print self.Z
		self.ax.plot_surface(xx, yy, self.Z, cmap=plt.cm.Spectral)

		# data = []
		# pred = []
		# for x in range(x_min, x_max, i):
		# 	for y in range(y_min, y_max, i):
		# 		for z in range(z_min, z_max, i):
		# 			data.append([x,y,z])
		self.ax.scatter(self.train_X[:,0], self.train_X[:,1], self.train_X[:,2], c=self.train_y, s=20, cmap=plt.cm.Spectral)
