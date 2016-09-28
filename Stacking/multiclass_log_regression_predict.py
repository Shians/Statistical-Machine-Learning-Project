import os
import numpy as np
from sklearn.linear_model import LogisticRegression

numBooks = 6

train_path = os.path.join(os.getcwd(), "Dummy_Model_Output.csv")
test_path = os.path.join(os.getcwd(), "Dummy_Test_Output.csv")

output_filename = "stack6_predict.csv"

# import output from each model for training data
with open(train_path, 'rb') as csvfile:
    dataTrain = np.loadtxt(csvfile, delimiter=',', skiprows = 1)

# import output from each model for test data
with open(test_path, 'rb') as csvfile:
    dataTest = np.loadtxt(csvfile, delimiter=',', skiprows = 1)

# prediction arrays
testIds = np.empty([0,0])
testPredict = np.empty([0,0])

for book in range(numBooks):
    # book data
    dataBook = dataTrain[dataTrain[:,1] == book,:]
    
    # fit multiclass logistic regression
    LogReg = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    LogReg.fit(dataBook[:,2:], dataBook[:,0])
    
    # predict test data
    testIds = np.append(testIds, dataTest[dataTest[:,1] == book, 0])
    testPredict = np.append(testPredict, LogReg.predict(dataTest[dataTest[:,1] == book, 2:]))

# delete output file if exists
if os.path.exists(output_filename):
    os.remove(output_filename)

# write output
output = np.column_stack((testIds, testPredict))
np.savetxt(output_filename, output, fmt='%d', header="Id,Character", delimiter=",", comments="")