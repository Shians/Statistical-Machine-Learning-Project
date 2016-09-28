import os
import numpy as np
from sklearn.linear_model import LogisticRegression

numBooks = 6

train_path = os.path.join(os.getcwd(), "Dummy_Model_Output.csv")

# import output from each model
with open(train_path, 'rb') as csvfile:
    dataImport = np.loadtxt(csvfile, delimiter=',', skiprows = 1)

# accuracy for each book
acc = [0]*numBooks

# overall training accuracy
trainAcc = 0

for book in range(numBooks):
    # book data
    dataBook = dataImport[dataImport[:,1] == book,:]
    
    # fit multiclass logistic regression
    LogReg = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    LogReg.fit(dataBook[:,2:], dataBook[:,0])
    
    # accuracy on training data
    acc[book] = LogReg.score(dataBook[:,2:], dataBook[:,0])
    print("Training Accuracy on Book " + str(book) + " : " + str(round(acc[book], 4)))

    # update overall training accuracy
    trainAcc += acc[book] * dataBook.shape[0] / dataImport.shape[0]

print("Overall Training Accuracy : " + str(round(trainAcc, 4)))