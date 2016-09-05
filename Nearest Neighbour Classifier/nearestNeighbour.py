import os
import numpy as np
from collections import Counter
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

full_train_path = os.path.join(os.getcwd(), "..", "data", "train.csv")
small_train_path = os.path.join(os.getcwd(), "..", "data", "smallTrain.csv")

useFull = True

if (useFull):
    # import full training data set
    with open(full_train_path, 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')
else:
    # import small training data set
    with open(small_train_path, 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')

# randomize seed to reproduce results
seed = np.random.randint(low = 1000, high = 9999, size = None)
np.random.seed(seed)

print('Seed is ' + str(seed))

# label with count > 1
outcomes = [val[0] for val in Counter(dataImport[:,1]).items() if val[1] > 1]

# keep instances where label has count > 1
keep = [i for i in range(dataImport.shape[0]) if dataImport[i,1] in outcomes]
dataSub = dataImport[keep,:]

# isolate stratified portion of data for training
trainPortion = 0.7
indicies = StratifiedShuffleSplit(dataSub[:,1], n_iter = 1, train_size = trainPortion, random_state = seed)

for trainIndex, testIndex in indicies:
    # portion of data for training
    dataTrain = dataSub[trainIndex, 9:439]
    targetTrain = dataSub[trainIndex, 1]
    
    # portion of data for testing
    dataTest = dataSub[testIndex, 9:439]
    targetTest = dataSub[testIndex, 1]

# model selection to choose number of neighbours and L2 or L3 norm
kVec = np.arange(1, 16, 1)
pVec = np.arange(1, 4, 1)

for p in pVec:
    for k in kVec:
        # fit nearest neighbours classifier
        neigh = KNeighborsClassifier(n_neighbors = k, p = p, weights = 'uniform', algorithm = 'auto')
        neigh.fit(dataTrain, targetTrain)

        # find accuracy on test data
        acc = neigh.score(dataTest, targetTest)
        print('Accuracy for p = ' + str(p) + ', k = ' + str(k) + ': ' + str(round(acc, 4)))