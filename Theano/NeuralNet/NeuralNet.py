#! /usr/bin/python2
# tutorial from wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np
import matplotlib.pyplot as plt


# #scikit-learn
# np.random.seed(0)

class Data():
	def __init__(self, data, label):
		self.d = data
		self.l = label

	def __len__(self):
		return len(self.d)

	def __str__(self):
		return ('data: %4s,\tlabel: %2d' % (self.d, self.l))

def calculateLoss(model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)

	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)

	data_loss += reg_lamda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1./num_examples * data_loss

def predict(model, x):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)

	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
	b1 = np.zeros((1,nn_hdim))
	W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
	b2 = np.zeros((1, nn_output_dim))

	model = {}

	e = epsilon
	for i in xrange(0, num_passes):

		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)

		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		delta3 = probs
		delta3[range(num_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)

		dW2 += reg_lamda * W2
		dW1 += reg_lamda * W1
		
		e -= e*.0005
		W1 += -e * dW1
		b1 += -e * db1
		W2 += -e * dW2
		b2 += -e * db2

		model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

		if print_loss and i%1000==0:
			print "Loss after iteration %i: %f" %(i, calculateLoss(model))

	return model

def plot_decision_boundary(pred_func):
	x_min, x_max = X[:,0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:,1].min() - .5, X[:, 1].max() + .5
	h = 0.1

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap=plt.cm.Spectral)

train = []
# train.append(Data([2.781,2.550],0))
# train.append(Data([1.465,2.362],0))
# train.append(Data([3.396,4.400],0))
# train.append(Data([1.388,1.850],0))
# train.append(Data([3.064,3.005],0))
# train.append(Data([1.5,0.5],0))
# train.append(Data([8,10],0))
# train.append(Data([7.627,2.759],1))
# train.append(Data([5.332,2.088],1))
# train.append(Data([6.922,1.771],1))
# train.append(Data([8.675,-0.242],1))
# train.append(Data([7.673,3.508],1))
# train.append(Data([11,9],2))
# train.append(Data([14,6],2))
# train.append(Data([14,14],3))
# train.append(Data([10,14],3))

# train.append(Data([0,0],4))
# train.append(Data([-2,-2],4))

# train.append(Data([2.781,2.550],0))
# train.append(Data([8,5],0))
# train.append(Data([9,5],0))
# train.append(Data([1.465,2.362],0))
# train.append(Data([3.396,4.400],0))
# train.append(Data([1.388,1.850],0))
# train.append(Data([3.064,3.005],0))
# train.append(Data([1.5,0.5],0))
# train.append(Data([7.627,2.759],1))
# train.append(Data([5.332,2.088],1))
# train.append(Data([6.922,1.771],1))
# train.append(Data([8.675,-0.242],1))
# train.append(Data([7.673,3.508],1))
# train.append(Data([11,9],1))
# train.append(Data([2,3],0))
# train.append(Data([1,10],0))
# train.append(Data([8,10],0))
# train.append(Data([3,6],0))
# train.append(Data([2,1],0))
# train.append(Data([10,7],1))
# train.append(Data([5,2],1))
# train.append(Data([9,1],1))
# train.append(Data([12,10],1))
# train.append(Data([7,3],1))
# X = []
# y = []
# for t in train:
# 	X.append([t.d[0], t.d[1]])
# 	y.append(t.l)

# X = np.array(X)
# y = np.array(y) 
A = (5*np.random.random_sample((100,2))) + (2*np.random.random_sample((1,2))-1)
B = (5*np.random.random_sample((100,2))) + (2*np.random.random_sample((1,2))-1) + 3
X = np.concatenate((A,B))
y = 100*[0]
y.extend(100*[1])
print len(X), len(y)

#X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 4

epsilon = 0.01
reg_lamda = 0.01

model = build_model(5, num_passes=40000, print_loss=True)
plot_decision_boundary(lambda x: predict(model, x))
plt.title("yea, what the thing said")

plt.show()