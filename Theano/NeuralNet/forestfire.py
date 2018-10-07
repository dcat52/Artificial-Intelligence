import pandas as pd
import NN
import numpy as np

df = pd.read_csv('./forestfires.csv')

#print df.head(2)

x_axis = df['temp']
y_axis = df['rain']
qlty = df['month']

train_X = [a for a in zip(list(x_axis), list(y_axis))]
train_y = list(qlty)

temp_y = []
for y in train_y:
	y2 = -1
	if y == "jan":
		y2 = 0
	elif y == "feb":
		y2 = 1
	elif y == "mar":
		y2 = 2
	elif y == "apr":
		y2 = 3
	elif y == "may":
		y2 = 4
	elif y == "jun":
		y2 = 5
	elif y == "jul":
		y2 = 6
	elif y == "aug":
		y2 = 7
	elif y == "sep":
		y2 = 8
	elif y == "oct":
		y2 = 9
	elif y == "nov":
		y2 = 10
	elif y == "dec":
		y2 = 11
	temp_y.append(y2)

train_y = temp_y
numlabels = len(set(train_y))
print "num labels: %d." % numlabels


tX = np.asarray(train_X)
ty = np.asarray(train_y)

a = NN.NN(tX, ty, input_dim=2, hidden_dim=2000, output_dim=numlabels, num_passes=20000, block=True, UID=1, buff=1.0, h=0.1)
