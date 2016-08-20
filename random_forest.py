import csv
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

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
forest = RandomForestClassifier(n_estimators=5, n_jobs=2, max_features=None)
forest.fit(fit_data, fit_target)

# Predict and check result on remaining data
forest_expected = target[fit_size:]
forest_predicted = forest.predict(data[fit_size:])
print("Classification report for classifier {}:\n{}\n".format(forest, metrics.classification_report(forest_expected, forest_predicted)))