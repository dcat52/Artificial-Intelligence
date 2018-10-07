#!/usr/bin/python2

import numpy as np
from random import randint, shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron(object):

	def __init__(self, train, test=None, learnRate=0.1, numEpochs=5):
		self.learnRate = learnRate
		self.numEpochs = numEpochs
		self.train = train
		self.test = test
		self.weights = [0.0 for i in range(len(train[0])+1)]
		self.Ws = []

	def predict(self, datum):
		activation = self.weights[0]

		for i in range(len(datum)):
			activation += self.weights[i+1] * datum.d[i]

		return 1.0 if activation >= 0.0 else 0.0

	def train_weights(self):

		for epoch in range(self.numEpochs):

			sumErr = 0.0
			shuffle(self.train)
			for datum in self.train:

				prediction = self.predict(datum)
				error = datum.l - prediction
				sumErr += error**2
				self.weights[0] += self.learnRate * error

				for i in range(len(datum)):
					self.weights[i+1] += self.learnRate * error * datum.d[i]

			print('> epoch=%d, error=%.3f' % (epoch, sumErr))
			self.test_weights()
			self.Ws.append(self.weights)
		return self.weights

	def test_weights(self):
		testCount = len(self.test)
		correct = 0
		err = []
		for datum in self.test:
			res = self.predict(datum)
			if res == datum.l:
				correct += 1
			else:
				err.append(datum)
		print 'percent accuracy: %.2f%%.' % (correct/float(testCount)*100.0)
		return err



class Data():
	def __init__(self, data, label):
		self.d = data
		self.l = label

	def __len__(self):
		return len(self.d)

	def __str__(self):
		return ('data: %4s,\tlabel: %2d' % (self.d, self.l))

# mn = 0
# mx = 100

# train = []
# for x in range(1000):
# 	f1 = randint(mn,mx)
# 	label = mn
# 	if f1 >= mx/2.0:
# 		label = 1

# 	train.append(Data([f1,0],label))

train = []
train.append(Data([2.781,2.550],0))
train.append(Data([1.465,2.362],0))
train.append(Data([3.396,4.400],0))
train.append(Data([1.388,1.850],0))
train.append(Data([3.064,3.005],0))
train.append(Data([1.5,0.5],0))
train.append(Data([7.627,2.759],1))
train.append(Data([5.332,2.088],1))
train.append(Data([6.922,1.771],1))
train.append(Data([8.675,-0.242],1))
train.append(Data([7.673,3.508],1))
train.append(Data([11,9],1))

test = []
test.append(Data([2,3],0))
test.append(Data([1,10],0))
test.append(Data([8,10],0))
test.append(Data([3,6],0))
test.append(Data([2,1],0))
test.append(Data([10,7],1))
test.append(Data([5,2],1))
test.append(Data([9,1],1))
test.append(Data([12,10],1))
test.append(Data([7,3],1))

for x in train:
	print x


print "--"

p = Perceptron(train,test, learnRate=0.1, numEpochs=10)
weights = p.train_weights()

print "---"

print weights

err = p.test_weights()

# test2 = []
# for x in range(13*4):
# 	for y in range(13*4):
# 		test2.append(Data([x/4.0,y/4.0],0))

# p.test = test2
# err2 = p.test_weights()

for dat in train:
	if(dat.l == 0):
		plt.plot(dat.d[0], dat.d[1], 'bo', markersize=15)
	else:
		plt.plot(dat.d[0], dat.d[1], 'ro', markersize=15)

for dat in test:
	if(dat.l == 0):
		plt.plot(dat.d[0], dat.d[1], 'cs', markersize=12)
	else:
		plt.plot(dat.d[0], dat.d[1], 'ms', markersize=12)

for dat in err:
	plt.plot(dat.d[0], dat.d[1], 'ko', markersize=9)
#plt.show()

# for dat in test2:
# 	if dat not in err2:
# 		plt.plot(dat.d[0], dat.d[1], 'bo', markersize=3)
# for dat in err2:
# 	plt.plot(dat.d[0], dat.d[1], 'ro', markersize=3)

# print p.Ws
# for weights in p.Ws:
# 	x = np.linspace(0, 12, 1000)
# 	plt.plot(x, -(weights[1]*x+weights[0])/weights[2], 'k', linewidth=2)
# # print(weights[1]*5+weights[0])

# fig = plt.figure()

# ax = fig.add_subplot(111,projection='3d')
# for dat in train:
# 	ax.scatter(dat.d[0], dat.d[1], dat.l, 'bo', s=100)
plt.show()