import csv
import numpy as np
from sklearn import datasets, svm, metrics

with open('data/train.csv', 'rb') as csvfile:
    training_data = np.loadtxt(csvfile, delimiter=',')

# Permute data
permuted_data = np.random.permutation(training_data)
n_samples = len(permuted_data)

# Extract bitmaps and character annotation
data = permuted_data[:, 9:439]
target = permuted_data[:, 1]

# Isolate portion of data for training
train_portion = 0.8
fit_size = int(n_samples * train_portion)
fit_data = data[:fit_size]
fit_target = target[:fit_size]

# Fit random forest
classifier = svm.SVC(gamma = 1e-6)
classifier.fit(fit_data, fit_target)

# Predict and check result on remaining data
expected = target[fit_size:]
predicted = classifier.predict(data[fit_size:])
print("Classification report for classifier {}:\n{}\n".format(classifier, metrics.classification_report(expected, predicted)))
