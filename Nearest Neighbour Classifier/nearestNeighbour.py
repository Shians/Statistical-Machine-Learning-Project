import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

useFull = True

if (useFull):
    # import full training data set
    with open('../data/train.csv', 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')
else:
    # import small training data set
    with open('../data/smallTrain.csv', 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')

# size of data set
numSamples = len(dataImport)

# randomize seed to reproduce results
seed = np.random.randint(low = 1000, high = 9999, size = None)
np.random.seed(seed)

print('Seed is ' + str(seed))

# permute data
dataPermute = np.random.permutation(dataImport)

# extract bitmaps and character annotation
data = dataPermute[:, 9:439]
target = dataPermute[:, 1]

# isolate portion of data for training
trainPortion = 0.8
trainSize = int(trainPortion * numSamples)

dataTrain = data[0:trainSize]
targetTrain = target[0:trainSize]

# portion of data for testing
dataTest = data[trainSize:]
targetTest = target[trainSize:]

# model selection to choose number of neighbours
kVec = np.arange(7, 12, 1)

for k in kVec:
    # fit nearest neighbours classifier (with L2-norm)
    neigh = KNeighborsClassifier(n_neighbors = k, weights = 'uniform', algorithm = 'auto', p = 2)
    neigh.fit(dataTrain, targetTrain)

    # find accuracy on test data
    acc = neigh.score(dataTest, targetTest)
    print('Accuracy for k = ' + str(k) + ' is ' + str(round(acc,4)))