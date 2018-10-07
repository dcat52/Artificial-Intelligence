#!/usr/bin/python2

import numpy as np
from random import randint, shuffle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import Counter


class PerceptronClassifier:

	def __init__(self, train=[], test=[], labels=[], features=[], learnRate=0.1, numEpochs=5):
		self.learnRate = learnRate
		self.numEpochs = numEpochs

		self.train = train
		self.test = test

		self.labels = labels
		self.weights = {}
		for label in self.labels:
			self.weights[label] = Counter()
		self.Ws = []
		self.Es = []

	def predict(self, datum):
		guesses = []

		vectors = Counter()
		for l in self.labels:
			vectors[l] = self.weights[l] * datum.features
		guesses.append(vectors.argMax())
		return guesses

	def train_weights(self):

		for epoch in range(self.numEpochs):
			self.ce = epoch

			sumErr = 0.0
			shuffle(self.train)

			for datum in self.train:

				prediction = self.predict(datum)[0]
				if prediction != datum.l:
					self.weights[datum.l] += datum.features
					self.weights[prediction] -= datum.features

			print('> epoch=%d, error=%.3f' % (epoch, sumErr))
			self.test_weights()
			self.Ws.append(self.weights)
		return self.weights

	def test_weights(self, printer=True):
		testCount = len(self.test)
		correct = 0
		err = []
		for datum in self.test:
			res = self.predict(datum)[0]
			if res == datum.l:
				correct += 1
			else:
				err.append(datum)
		if printer:
			print 'percent accuracy: %.2f%%.' % (correct/float(testCount)*100.0)

		#f2 = plt.figure(figsize=(12,9))
		self.Es .append(err)
		return err


class XYData():
	def __init__(self, data, label):
		self.d = data
		self.l = label
		self.getFeatures()

	def __len__(self):
		return len(self.d)

	def __str__(self):
		return ('data: %4s,\tlabel: %2s' % (self.d, self.l))	

	def getFeatures(self):
		self.features = Counter()
		self.features['x-val'] = self.d[0]
		self.features['y-val'] = self.d[1]
		# self.features['x*y'] = self.d[0]*self.d[1]
		self.features['bias'] = 1





labels=['B','R','G']

train = []
Data = XYData
train.append(Data([2.781,2.550],'B'))
train.append(Data([1.465,2.362],'B'))
train.append(Data([3.396,4.400],'B'))
train.append(Data([1.388,1.850],'B'))
train.append(Data([3.064,3.005],'B'))
train.append(Data([1.5,0.5],'B'))
train.append(Data([7.627,2.759],'R'))
train.append(Data([5.332,2.088],'R'))
train.append(Data([6.922,1.771],'R'))
train.append(Data([8.675,-0.242],'R'))
train.append(Data([7.673,3.508],'R'))
train.append(Data([11,9],'G'))

test = []
test.append(Data([2,3],'B'))
test.append(Data([1,10],'B'))
test.append(Data([8,10],'G'))
test.append(Data([3,6],'B'))
test.append(Data([2,1],'B'))
test.append(Data([10,7],'G'))
test.append(Data([5,2],'R'))
test.append(Data([9,1],'R'))
test.append(Data([12,10],'G'))
test.append(Data([7,3],'R'))

for x in train:
	print x

print "--"

p = PerceptronClassifier(train,test, learnRate=0.1, numEpochs=20, labels=labels)
p.train_weights()

print "---"

err = p.test_weights()

x = []
y = []
for e in range(len(p.Es)):
	x.append(e+1)
	y.append(len(p.Es[e]))
plt.plot(x, y, '-o', mew=3, markersize=1)
plt.show(block=False)

plt.figure(figsize=(12,9))
plt.xlim(-1, 12.5)
plt.ylim(-1, 12.5)

for dat in err:
	plt.plot(dat.d[0], dat.d[1], 'ws', mew=3, markersize=15)

for dat in train:
	if(dat.l == 'B'):
		plt.plot(dat.d[0], dat.d[1], color=[0,0,1], marker='o', ms=30)
	elif(dat.l == 'G'):
		plt.plot(dat.d[0], dat.d[1], color='g', marker='o', markersize=30)
	else:
		plt.plot(dat.d[0], dat.d[1], color=[1,0,0], marker='o', markersize=30)

for dat in test:
	if(dat.l == 'B'):
		plt.plot(dat.d[0], dat.d[1], color=[0,0,1], marker='*', markersize=30)
	elif(dat.l == 'G'):
		plt.plot(dat.d[0], dat.d[1], color='g', marker='*', markersize=30)
	else:
		plt.plot(dat.d[0], dat.d[1], color=[1,0,0], marker='*', markersize=30)

test2 = []
for x in range(-3,13*4):
	for y in range(-3,13*4):
		test2.append(Data([x/4.0,y/4.0],'B'))
p.test = test2
err2 = p.test_weights(printer=False)

for d in test2:
	d.l = 'R'
err3 = p.test_weights(printer=False)

r = []
g = []
b = []

g = list(set(err2) & set(err3))
r = list(set(err2) - set(err3))
b = list(set(err3) - set(err2))

for dat in r:
	plt.plot(dat.d[0], dat.d[1], color=[1,0,0], marker='o', mew= 0, ms=10)
for dat in g:
	plt.plot(dat.d[0], dat.d[1], color='g', marker='o', mew= 0, ms=10)
for dat in b:
	plt.plot(dat.d[0], dat.d[1], color=[0,0,1], marker='o', mew= 0, ms=10)

print p.weights

# for l in labels:
# 	weights = p.weights[l]
# 	x = np.linspace(0, 12, 1000)
# 	x1 = weights['x-val']
# 	y1 = weights['y-val']
# 	x2 = 0
# 	y2 = weights['bias']
# 	m = (y2-y1)/(x2-x1)
# 	print m
# 	plt.plot(x, (x2*x)/y1+y2, 'k', linewidth=2)


plt.get_current_fig_manager().window.wm_geometry("+2000+1000")
plt.show(block=False)
_ = raw_input()