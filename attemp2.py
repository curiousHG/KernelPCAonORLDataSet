from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import LabelEncoder
from utils.datasetMake import get_dataset
from utils.testTrainSplit import testTrainSplit


X, y = get_dataset()

le = LabelEncoder()
le.fit_transform(y)
# print(le.classes_)
trainX, trainY, testX, testY = testTrainSplit(X, y, 1)
print(len(trainX))
