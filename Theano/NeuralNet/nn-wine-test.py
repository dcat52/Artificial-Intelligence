import pandas as pd
import NN
import numpy as np

df = pd.read_csv('./winequality-red.csv')

#print df.head(2)

x_axis = df['sulphates']
y_axis = df['citric acid']
qlty = df['quality']

train_X = [a for a in zip(list(x_axis), list(y_axis))]
train_y = list(qlty)
minlab = min(train_y)
print minlab
print train_y.count(3),train_y.count(4),train_y.count(5),train_y.count(6),train_y.count(7),train_y.count(8)
train_y = [(y-minlab)/3 for y in train_y]
numlabels = len(set(train_y))
print numlabels
print train_y.count(0),train_y.count(1)


tX = np.asarray(train_X)
ty = np.asarray(train_y)

a = NN.NN(tX, ty, input_dim=2, hidden_dim=2000, output_dim=2, num_passes=2000, block=True, UID=1, buff=0.01, h=0.01)
