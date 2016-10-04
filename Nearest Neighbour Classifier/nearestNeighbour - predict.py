import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_path = os.path.join(os.getcwd(), "..", "data", "train.csv")
test_path = os.path.join(os.getcwd(), "..", "data", "test.csv")

# import full training data set
with open(train_path, 'rb') as csvfile:
    dataTrainImport = np.loadtxt(csvfile, delimiter=',')

# import full test data set
with open(test_path, 'rb') as csvfile:
    dataTestImport = np.loadtxt(csvfile, delimiter=',')

# extract bitmaps and character annotation
dataTrain = dataTrainImport[:, 9:439]
targetTrain = dataTrainImport[:, 1]

dataTest = dataTestImport[:, 9:439]
idsTest = dataTestImport[:, 0]

# number of neighbours from cross validation
k = 15

# norm from cross validation
p = 3

# fit nearest neighbours classifier
neigh = KNeighborsClassifier(n_neighbors = k, p = p, weights = 'uniform', algorithm = 'auto')
neigh.fit(dataTrain, targetTrain)

# predict test data
testPredict = neigh.predict(dataTest)

# delete output file if exists
if os.path.exists('nearestNeighbourPredict.csv'):
    os.remove('nearestNeighbourPredict.csv')

# write output
output = np.column_stack((idsTest, testPredict))
np.savetxt('nearestNeighbourPredict.csv', output, fmt='%d', header="Id,Character", delimiter=",", comments="")