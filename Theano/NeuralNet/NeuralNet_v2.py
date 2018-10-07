#! /usr/bin/python2
# tutorial from wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

class Data():
	def __init__(self, data, label):
		self.d = data
		self.l = label

	def __len__(self):
		return len(self.d)

	def __str__(self):
		return ('data: %4s,\tlabel: %2d' % (self.d, self.l))

def build_model(num_passes=20000, print_loss=False):
	np.random.seed(0)
	W1.set_value((np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)).astype('float32'))
	b1.set_value(np.zeros(nn_hdim).astype('float32'))
	W2.set_value((np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)).astype('float32'))
	b2.set_value(np.zeros(nn_output_dim).astype('float32'))

	for i in xrange(0, num_passes):

		gradient_step()

		if print_loss and i%1000==0:
			print "Loss after iteration %i: %f" %(i, calculate_loss())

def plot_decision_boundary(pred_func):
	x_min, x_max = train_X[:,0].min() - .5, train_X[:, 0].max() + .5
	y_min, y_max = train_X[:,1].min() - .5, train_X[:, 1].max() + .5
	h = 0.1

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	X.set_value((np.c_[xx.ravel(), yy.ravel()]).astype('float32'))
	Z = pred_func()

	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(train_X[:,0], train_X[:,1], c=train_y, s=20, cmap=plt.cm.Spectral)



import sklearn.datasets
np.random.seed(0)
#train_X, train_y = sklearn.datasets.make_moons(5000, noise=0.20)
A = (5*np.random.random_sample((100,2))) + (2*np.random.random_sample((1,2))-1)
B = (5*np.random.random_sample((100,2))) + (2*np.random.random_sample((1,2))-1) + 3
train_X = np.concatenate((A,B))
train_y = 100*[0]
train_y.extend(100*[1])

train_y = np.asarray(train_y)

train_X = train_X.astype(np.float32)
train_y = train_y.astype(np.int32)
train_y_onehot = np.eye(2)[train_y]

# A = (5*np.random.random_sample((100,2))) + (2*np.random.random_sample((1,2))-1)
# B = (5*np.random.random_sample((100,2))) + (2*np.random.random_sample((1,2))-1) + 3
# C = np.concatenate((A,B))
# d = 100*[0]
# d.extend(100*[1])
# print len(C),len(d)

X = theano.shared(train_X.astype('float32'))
y = theano.shared(train_y_onehot.astype('float32'))

num_examples = len(train_X)
nn_input_dim = 2
nn_hdim = 10
nn_output_dim = 2
num_passes = 10000

epsilon = np.float32(0.01)
reg_lambda = np.float32(0.01)

W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim).astype('float32'), name='W1')
b1 = theano.shared(np.zeros(nn_hdim).astype('float32'), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim).astype('float32'), name='W2')
b2 = theano.shared(np.zeros(nn_hdim).astype('float32'), name='b2')

z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)

loss_reg = 1./num_examples * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
loss = T.nnet.categorical_crossentropy(y_hat, train_y).mean() + loss_reg

prediction = T.argmax(y_hat, axis=1)

dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

forward_prop = theano.function([], y_hat)
calculate_loss = theano.function([], loss)
predict = theano.function([], prediction)

gradient_step = theano.function([], updates=(
	(W2, W2 - epsilon * dW2),
	(W1, W1 - epsilon * dW1),
	(b2, b2 - epsilon * db2),
	(b1, b1 - epsilon * db1)))

build_model(num_passes, print_loss=True)
plot_decision_boundary(predict)
plt.title("%d classes - %d hidden layer width - %d iterations" % (nn_output_dim, nn_hdim, num_passes))
plt.show(block=True)

# W1.set_value(np.random.randn(nn_input_dim, nn_hdim).astype('float32'))
# W2.set_value(np.random.randn(nn_hdim, nn_output_dim).astype('float32'))
# b1.set_value(np.zeros(nn_hdim).astype('float32'))
# b2.set_value(np.zeros(nn_hdim).astype('float32'))
# dW2 = T.grad(loss, W2)
# db2 = T.grad(loss, b2)
# dW1 = T.grad(loss, W1)
# db1 = T.grad(loss, b1)
# nn_input_dim = 2
# nn_hdim = 5
# nn_output_dim = 2
# num_passes = 50000
# build_model(num_passes, print_loss=True)
# plot_decision_boundary(predict)
# plt.title("%d classes - %d hidden layer width - %d iterations" % (nn_output_dim, nn_hdim, num_passes))
# plt.show()