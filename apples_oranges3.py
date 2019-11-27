
from sklearn.svm import SVC
import numpy as np
import csv


def csv2X(path, split=False):
	with open(path, 'r') as f:
		return  np.array(list(csv.reader(f,delimiter=','))).astype(float)

def csv2y(path, split=False):
	with open(path, 'r') as f:
		return  np.array(list(csv.reader(f,delimiter=','))).astype(float).ravel()


features = csv2X('datasets/fruit_data_binaryX.csv')
labels = csv2y('datasets/fruit_data_binaryY.csv')

# svclassifier = SVC(kernel='rbf', gamma='auto')
svclassifier = SVC(kernel='linear')

svclassifier.fit(features, labels)


print (svclassifier.predict([[154,7.1,7.5]]))

