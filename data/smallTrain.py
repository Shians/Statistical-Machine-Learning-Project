import os
import numpy as np
from collections import Counter
from sklearn.cross_validation import StratifiedShuffleSplit

# read full training set
with open('train.csv', 'rb') as csvfile:
    dataAll = np.loadtxt(csvfile, delimiter=',')
    
# randomize seed to reproduce results
seed = np.random.randint(low = 1000, high = 9999, size = None)

print('Seed is ' + str(seed))

# label with count > 1
outcomes = [val[0] for val in Counter(dataAll[:,1]).items() if val[1] > 1]

# keep instances where label has count > 1
keep = [i for i in range(dataAll.shape[0]) if dataAll[i,1] in outcomes]
dataSub = dataAll[keep,:]

# stratified sample
indicies = StratifiedShuffleSplit(dataSub[:,1], n_iter = 1, train_size = 1000 / dataSub.shape[0], random_state = seed)

# small training data with about 1000 observations
dataSmall = dataSub[[train_index for train_index, test_index in indicies][0],:]

# delete file if exists
if os.path.exists("smallTrain.csv"): os.remove("smallTrain.csv")

# save as csv
np.savetxt('smallTrain.csv', dataSmall, fmt='%d', delimiter=",", comments="")