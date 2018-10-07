from NN import NN


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import uniform
import time

np.random.seed(0)

random.seed(0)
A,B,C = [],[],[]

num = 40

for i in range(0,num):
	A.append([uniform(0,3),uniform(0,2.5)])
	B.append([uniform(2.0,6),uniform(1.5,3)])
	C.append([uniform(5,7),uniform(0,2.5)])

A = 2*np.asarray(A)
B = 2*np.asarray(B)
C = 2*np.asarray(C)

train_X = np.concatenate((A,B))
train_X = np.concatenate((train_X,C))
train_y = num*[0]
train_y.extend(num*[1])
train_y.extend(num*[2])
train_y = np.asarray(train_y)

class obj:
	def __init__(self, xx,yy,Z,nn_hdim):
		self.xx = xx
		self.yy = yy
		self.Z = Z
		self.nn_hdim = nn_hdim
nets = []
passes = 10000
hidden = 100
for i in range(1):
	print i
	a = NN(train_X, train_y, input_dim=2, hidden_dim=int(hidden), output_dim=3, num_passes=int(passes), block=False, UID=1, h=0.05)
	#b = obj(a.xx,a.yy,a.Z,a.nn_hdim)
	#del a
	nets.append(a)
	#passes += 100
	passes += 1000

print "About to play!!!"
#time.sleep(5)
# a = NN(train_X, train_y, input_dim=2, hidden_dim=int(1E2), output_dim=classes, num_passes=int(1E0), block=False, UID=1, h=0.05)
# b = NN(train_X, train_y, input_dim=2, hidden_dim=int(1E2), output_dim=classes, num_passes=int(1E1), block=False, UID=2, h=0.05)
# c = NN(train_X, train_y, input_dim=2, hidden_dim=int(1E2), output_dim=classes, num_passes=int(1E2), block=False, UID=3, h=0.05)
# d = NN(train_X, train_y, input_dim=2, hidden_dim=int(1E2), output_dim=classes, num_passes=int(1E3), block=False, UID=4, h=0.05)
# e = NN(train_X, train_y, input_dim=2, hidden_dim=int(1E2), output_dim=classes, num_passes=int(1E4), block=False, UID=5, h=0.05)
# f = NN(train_X, train_y, input_dim=2, hidden_dim=int(1E2), output_dim=classes, num_passes=int(1E5), block=True, UID=6, h=.05)

for i,v in enumerate(nets):
	#print i
	plt.figure(0)
	plt.title("%d classes - %d hidden layer width - %d iterations" % (v.nn_output_dim, v.nn_hdim, v.num_passes))
	plt.contourf(v.xx, v.yy, v.Z, cmap=plt.cm.Spectral)
	plt.show(block=False)
	plt.scatter(v.train_X[:,0], v.train_X[:,1], c=v.train_y, s=20, cmap=plt.cm.Spectral)
	plt.show(block=False)
	plt.pause(0.01)
	time.sleep(0.03)

plt.show()
