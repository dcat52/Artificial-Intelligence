import pandas as pd
import NN
import numpy as np

df = pd.read_csv('/home/davis/Documents/CanonVA/AI/NeuralNet/Absenteeism_at_work.csv')

# print df.head(2)

x_axis = df['Transportation expense']
y_axis = df['Absenteeism time in hours']
qlty = df['Seasons']

train_X = [a for a in zip(list(x_axis), list(y_axis))]
train_y = list(qlty)

train_y = [y-1 for y in train_y]
numlabels = len(set(train_y))
print "num labels: %d." % numlabels

tX = np.asarray(train_X)
ty = np.asarray(train_y)

print "%d data points!" % len(train_y)
a = NN.NN(tX, ty, input_dim=2, hidden_dim=100, output_dim=numlabels, num_passes=10000, block=True, UID=1, buff=1.0, h=0.1)
