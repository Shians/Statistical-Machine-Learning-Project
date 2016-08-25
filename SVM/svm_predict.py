import csv
import numpy as np
from sklearn import datasets, svm, metrics

# Read in data
with open('../data/train.csv', 'rb') as csvfile:
    training_data = np.loadtxt(csvfile, delimiter=',')

with open('../data/test.csv', 'rb') as csvfile:
    test_data = np.loadtxt(csvfile, delimiter=',')

# Define classifier
classifier = svm.SVC(gamma = 1e-6)

# Train model
data = training_data[:, 9:439]
target = training_data[:, 1]
pclassifier = svm.SVC(gamma = 1e-6)
classifier.fit(data, target)

# Prediction
ids = test_data[:, 0]
test_data_bitmaps = test_data[:, 9:439]
predicted = classifier.predict(test_data_bitmaps)

# Write output
output = np.column_stack((ids, predicted))
np.savetxt('svm_prediction.csv', output, 
				fmt='%d', header="Id,Character", delimiter=",", comments="")