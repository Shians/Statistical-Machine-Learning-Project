import os
import numpy as np
from sklearn.linear_model import LogisticRegression

numBooks = 6

train_model_path = os.path.join(os.getcwd(), "Logistic_Input", "Training_Probs.csv")
test_model_path = os.path.join(os.getcwd(), "Logistic_Input", "Testing_Probs.csv")
test_input_path = os.path.join(os.getcwd(), "..", "data", "test.csv")

output_filename = "stack6_predict.csv"

# import output from each model for training data
with open(train_model_path, 'rb') as csvfile:
    dataModelTrain = np.loadtxt(csvfile, delimiter=',')

print('Read in Training Data')

# import output from each model for test data
with open(test_model_path, 'rb') as csvfile:
    dataModelTest = np.loadtxt(csvfile, delimiter=',')

print('Read in Test Data')

# import test data to obtain test ids
with open(test_input_path, 'rb') as csvfile:
    dataInputTest = np.loadtxt(csvfile, delimiter=',')

# load test ids
testIds = dataInputTest[:,0]

print('Read in Test Ids\n')

# prediction array
testPredict = np.empty([0,0])

# training accuracy for each book
acc = [0]*numBooks

# overall training accuracy
trainAcc = 0

for book in range(numBooks):
    # book data
    dataBook = dataModelTrain[dataModelTrain[:,1] == book,:]
    
    # fit multiclass logistic regression
    LogReg = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    LogReg.fit(dataBook[:,2:], dataBook[:,0])
    
    # accuracy on training data
    acc[book] = LogReg.score(dataBook[:,2:], dataBook[:,0])
    print("Training Accuracy on Book " + str(book) + " : " + str(round(acc[book], 4)))

    # update overall training accuracy
    trainAcc += acc[book] * dataBook.shape[0] / dataModelTrain.shape[0]

    # predict test data
    testPredict = np.append(testPredict, LogReg.predict(dataModelTest[dataModelTest[:,1] == book, 2:]))

print("Overall Training Accuracy : " + str(round(trainAcc, 4)))

# delete output file if exists
if os.path.exists(output_filename):
    os.remove(output_filename)

# write output
output = np.column_stack((testIds, testPredict))
np.savetxt(output_filename, output, fmt='%d', header="Id,Character", delimiter=",", comments="")