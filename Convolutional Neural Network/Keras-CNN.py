import os
import numpy as np

from fitCNN import fitCNN

full_train_path = os.path.join(os.getcwd(), "..", "data", "train.csv")
small_train_path = os.path.join(os.getcwd(), "..", "data", "smallTrain.csv")

useFull = False

if (useFull):
    # import full training data set
    with open(full_train_path, 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')
else:
    # import small training data set
    with open(small_train_path, 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')

perm_data = np.random.permutation(dataImport)
n_samples = len(perm_data)

# input image dimensions
img_rows, img_cols = 33, 13

data = perm_data[:, 9:439].reshape(n_samples, 1, img_rows, img_cols)
target = perm_data[:, 1].astype(int)

fitCNN(data, target, modelName = 'model_small.h5')