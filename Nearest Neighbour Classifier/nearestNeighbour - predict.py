import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# import full training data set
with open('../data/train.csv', 'rb') as csvfile:
    dataTrainImport = np.loadtxt(csvfile, delimiter=',')

# import full test data set
with open('../data/test.csv', 'rb') as csvfile:
    dataTestImport = np.loadtxt(csvfile, delimiter=',')

# extract bitmaps and character annotation
dataTrain = dataTrainImport[:, 9:439]
targetTrain = dataTrainImport[:, 1]

dataTest = dataTestImport[:, 9:439]
idsTest = dataTestImport[:, 0]

# number of neighbours from model selection
k = 9

# fit nearest neighbours classifier
neigh = KNeighborsClassifier(n_neighbors = k, weights = 'uniform', algorithm = 'auto')
neigh.fit(dataTrain, targetTrain)

# predict test data
testPredict = neigh.predict(dataTest)

# write output
output = np.column_stack((idsTest, testPredict))
np.savetxt('nearestNeighbourPredict.csv', output, fmt='%d', header="Id,Character", delimiter=",", comments="")