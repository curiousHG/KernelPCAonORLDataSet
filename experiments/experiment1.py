import sys

sys.path.append("../")

from utils.datasetMake import get_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.testTrainSplit import testTrainSplit

X, y = get_dataset()

scalar = StandardScaler()
X = scalar.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, Y_train, X_test, Y_test = testTrainSplit(X, y, 1)

from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

nComps = [10, 20, 30]
widths = [i for i in range(200, 4401, 200)]

comps = []

for components in nComps:
	vals = []
	for width in widths:
		# KernelPCA to the input data
		# 𝜅(𝐮,𝐯)=exp(−𝛾‖𝐮−𝐯‖2)
		# width = σ squared
		gama = 1 / (2 * width)
		kPCA = KernelPCA(n_components=components, kernel="rbf", gamma=gama)
		new_X_train = kPCA.fit_transform(X_train)
		new_X_test = kPCA.transform(X_test)

		# CLASSIFICATION STEP
		# metric = minkowski, p=2 makes it euclidean distance
		classifier = KNeighborsClassifier(2, metric="minkowski", p=2)
		classifier.fit(new_X_train, Y_train)
		y_pred = classifier.predict(new_X_test)

		# cm = confusion_matrix(Y_test,y_pred)
		# print(classification_report(y_true=Y_test,y_pred=y_pred))
		# print(classification_report(y_true=le.inverse_transform(Y_test),y_pred=le.inverse_transform(y_pred),zero_division=))
		# print(int(accuracy_score(y_true=Y_test, y_pred=y_pred) * 100), f'Components = {components} and width = {width}')
		vals.append(accuracy_score(y_true=Y_test, y_pred=y_pred) * 100)
	comps.append(vals)


import matplotlib.pyplot as plt

for i in range(3):
	plt.plot(widths, comps[i], label=nComps[i])
plt.title("Gaussian KernelPCA using varying width and number of components")
plt.legend()
plt.grid(True)
plt.ylim([0,80])
plt.xlim([100,4500])
plt.show()
