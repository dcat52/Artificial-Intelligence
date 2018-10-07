#! /usr/bin/python2
# tutorial from deeplearning.net/tutorial/mlp.html

from __future__ import print_function
__docformat__ = 'restructuredtext en'
import os, sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

class Layer(object):
	def __init__(self, n_in, n_out, layer_width=100, W=None, b=None, activation=T.tanh):
		self.n_in = n_in
		self.n_out = n_out
		self.layer_width = layer_width

		self.W = theano.shared(np.random.randn(self.n_in, self.n_out).astype('float32'))
		self.b = theano.shared(np.zeros(self.layer_width).astype('float32'))

class MLP(object):
	def __init__(self, data, classes):
		self.X = data[0]
		self.y = data[1]

		self.classes = classes
		self.num_examples = len(self.data)

		self.num_epochs = 1000

		self.epsilon = np.float32(0.01)
		self.reg_lambda = np.float32(0.01)

		self.hl = Layer(
			n_in=2,
			n_out=2,
			layer_width=100
			)

		self.logRegressionLayer = LogisticRegression(
			input=self.HiddenLayer.output,
			n_in=n_hidden,
			n_out=n_out
			)

		self.L1 = (
			abs(self.HiddenLayer.W).sum()
			+ abs(self.logRegressionLayer.W).sum()
			)

		self.L2_sqr = (
			(self.HiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
			)

		self.L2_sqr = (
			(self.HiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
			)

		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
			)

		self.z1 = self.X.dot(hl.W) + hl.b
		self.a1 = T.tanh(self.z1)
		self.z2 = self.a1.dot(R.W) + R.b
		self.y_hat = T.nnet.softmax(self.z2)

		self.loss_reg = 1./self.num_examples * self.reg_lambda/2 * (T.sum(T.sqr(L1.W)) + T.sum(T.sqr(R.W)))
		self.