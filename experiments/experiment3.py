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
trX, trY, tsX, tsY = testTrainSplit(X, y, 1)

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

components = [10, 20, 30, 40]
accuracy_components = []

for comp in components:
    transformer = PCA(n_components=comp)
    X_train = transformer.fit_transform(trX)
    X_test = transformer.transform(tsX)

    neigh = KNeighborsClassifier(n_neighbors=1, p=2)
    neigh.fit(X_train, trY)

    y_pred = neigh.predict(X_test)
    accuracy = accuracy_score(y_true=tsY, y_pred=y_pred) * 100
    accuracy_components.append(accuracy)

for j in range(len(accuracy_components)):
    print(f'Correct Recognition Rate at {components[j]}number of components is {accuracy_components[j]}')


import matplotlib.pyplot as plt

plt.plot(components, accuracy_components, label=comp)
plt.title("PCA using varying number of components")
plt.grid(True)
plt.xlabel("Number of Components")
plt.ylabel("Correct Recognition Rate")
plt.axhline(y=70.5, color='r', linestyle='dashed')
plt.annotate("Constant after threshold components", xy=(15, 71))
plt.ylim([0, 80])
plt.show()
