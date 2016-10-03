import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

full_train_path = os.path.join(os.getcwd(), "..", "data", "train.csv")
small_train_path = os.path.join(os.getcwd(), "..", "data", "smallTrain.csv")

useFull = True

if (useFull):
    # import full training data set
    with open(full_train_path, 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')
else:
    # import small training data set
    with open(small_train_path, 'rb') as csvfile:
        dataImport = np.loadtxt(csvfile, delimiter=',')

# param_grid = {'n_neighbors' : np.arange(1, 16, 1), 'p' : np.arange(1, 4, 1)}
param_grid = {'n_neighbors' : np.arange(16, 26, 1), 'p' : np.arange(1, 4, 1)}

neigh = KNeighborsClassifier(weights = 'uniform', algorithm = 'auto')

results = GridSearchCV(estimator = neigh, param_grid = param_grid, n_jobs = 15, cv = 5,
                       refit = False, verbose = 3)
results.fit(X = dataImport[:,9:439], y = dataImport[:,1])

numNeighbors = [val.parameters['n_neighbors'] for val in results.grid_scores_]
pNorm = [val.parameters['p'] for val in results.grid_scores_]
scores = [val.mean_validation_score for val in results.grid_scores_]

for i in range(len(numNeighbors)):
	with open('nearestNeighbour_CV.csv', 'a') as outputFile:
		outputFile.write(str(numNeighbors[i]) + "," + str(pNorm[i]) + "," + str(scores[i]) + "\n")

# # delete output file if exists
# if os.path.exists('nearestNeighbour_CV.csv'):
#     os.remove('nearestNeighbour_CV.csv')
    
# output = np.column_stack((numNeighbors, pNorm, scores))
# np.savetxt('nearestNeighbour_CV.csv', output, fmt='%f',
#            header="n_neighbors,p,cross_validation_accuracy", delimiter=",", comments="")
