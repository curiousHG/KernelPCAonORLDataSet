import numpy as np


def testTrainSplit(X, y, n):
	trainX = []
	trainY = []
	testX = []
	testY = []
	a = len(X[0])
	given = [0 for i in range(40)]
	for i in range(len(y)):
		if given[y[i]] < n:
			trainX.append(X[i])
			trainY.append(y[i])
			given[y[i]] += 1
		else:
			testX.append(X[i])
			testY.append(y[i])
	trainX = np.array(trainX, dtype=float)
	trainY = np.array(trainY,dtype=int)
	testX = np.array(testX, dtype=float)
	testY = np.array(testY,dtype=int)
	return trainX, trainY, testX, testY
