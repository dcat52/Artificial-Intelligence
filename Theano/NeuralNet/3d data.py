import NN
import numpy as np

import random
from random import uniform


np.random.seed(0)

random.seed(0)
A,B,C,D = [],[],[],[]

for i in range(0,150):
	A.append([uniform(0,2.8),uniform(0,3),uniform(0,3)])
	B.append([uniform(2.2,5),uniform(0,3),uniform(0,3)])

A = 2*np.asarray(A)
B = 2*np.asarray(B)

train_X = np.concatenate((A,B))

train_y = 150*[0]
train_y.extend(150*[1])
train_y = np.asarray(train_y)

a = NN.NN3D(train_X, train_y, input_dim=3, hidden_dim=100, output_dim=2, num_passes=1000, block=True, UID=1, buff=0.2, h=.1)
