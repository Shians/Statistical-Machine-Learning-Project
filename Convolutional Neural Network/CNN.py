import os
import numpy as np

from keras.models import load_model
from keras.utils import np_utils

test_path = os.path.join('..', 'data', 'test.csv')
with open(test_path, 'rb') as csvfile:
    testing_data = np.loadtxt(csvfile, delimiter=',')

model = load_model('model.h5')

img_rows, img_cols = 33, 13
# Bitmaps are rescaled to between 0-1 rather than 0-255, since model was trained that way.
test_bitmaps = testing_data[:, 9:439].reshape(testing_data.shape[0], 1, img_rows, img_cols)
test_bitmaps /= 255
test_ids = testing_data[:, 0]

test_predict = np_utils.categorical_probas_to_classes(model.predict(test_bitmaps))

output = np.column_stack((test_ids, test_predict))
np.savetxt('kerasCNN_Preprocessed.csv', output, fmt='%d', header="Id,Character", delimiter=",", comments="")