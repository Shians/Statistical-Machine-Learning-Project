import csv
import numpy as np

# read full training set
with open('train.csv', 'rb') as csvfile:
    dataAll = np.loadtxt(csvfile, delimiter=',')
    
# randomize seed to reproduce results
seed = np.random.randint(low = 1000, high = 9999, size = None)
np.random.seed(seed)

print('Seed is ' + str(seed))

# permute data
dataPermute = np.random.permutation(dataAll)

# select random 1000 instances
dataSmall = dataPermute[0:1000, :]

# save as csv
np.savetxt('smallTrain.csv', dataSmall, fmt='%d', delimiter=",", comments="")