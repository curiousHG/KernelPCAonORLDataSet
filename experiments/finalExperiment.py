import sys

sys.path.append("../")
import numpy as np
from utils.datasetMake import get_dataset
from utils.testTrainSplit import testTrainSplit
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as acc

X, Y = get_dataset()

scalar = StandardScaler()
X = scalar.fit_transform(X)

le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)

no_of_images = [int(i) for i in range(1, 10, 1)]

gaussian_width = 3000
components = 30
gaussian, polynomial, pca = [], [], []

for num in no_of_images:
    trX, trY, tsX, tsY = testTrainSplit(X, Y, num)

    # for rbf
    gamma = 1 / (2 * gaussian_width)
    transformer = KernelPCA(n_components=components, kernel='rbf', gamma=gamma)
    X_train = transformer.fit_transform(trX)
    X_test = transformer.transform(tsX)

    nn_rbf = KNeighborsClassifier(n_neighbors=1, p=2)
    nn_rbf.fit(X_train, trY)
    Y_pred = nn_rbf.predict(X_test)
    accuracy = acc(tsY, Y_pred)
    gaussian.append(accuracy * 100)

    # for poly
    poly_transformer = KernelPCA(n_components=components, kernel='poly', degree=2)
    X_train_poly = poly_transformer.fit_transform(trX)
    X_test_poly = poly_transformer.transform(tsX)

    nn_poly = KNeighborsClassifier(n_neighbors=1, p=2)
    nn_poly.fit(X_train_poly, trY)
    Y_pred_poly = nn_poly.predict(X_test_poly)
    accuracy = acc(tsY, Y_pred_poly)
    polynomial.append(accuracy * 100)

    # for PCA
    pca_transformer = PCA(n_components=components)
    X_train_pca = pca_transformer.fit_transform(trX)
    X_test_pca = pca_transformer.transform(tsX)

    nn_pca = KNeighborsClassifier(n_neighbors=1, p=2)
    nn_pca.fit(X_train_pca, trY)
    Y_pred_pca = nn_pca.predict(X_test_pca)
    accuracy = acc(tsY, Y_pred_pca)
    pca.append(accuracy * 100)

meanAccuracy = [(gaussian[i] + polynomial[i] + pca[i]) / 3 for i in range(len(pca))]

import matplotlib.pyplot as plt

x = np.arange(1, len(no_of_images) + 1)
plt.xticks(x, no_of_images)
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.plot(no_of_images, meanAccuracy,color = 'orange')
ax = plt.subplot()
ax.bar(no_of_images, gaussian, width=0.2, color='g', align='center')
ax.bar(np.array(no_of_images) - 0.2, polynomial, width=0.2, color='b', align='center')
ax.bar(np.array(no_of_images) + 0.2, pca, width=0.2, color='r', align='center')
ax.yaxis.grid(True)
ax.legend(['Mean accuracy','KPCA(Gaussian width 3000)', 'KPCA(Polynomial with degree 2)', 'PCA'])
plt.title("Performance results with varying number of training images per person (using 30 features)")
plt.xlabel("Number of Training Images")
plt.ylabel("Correct Recognition Rate")
plt.ylim([0, 100])
plt.show()
