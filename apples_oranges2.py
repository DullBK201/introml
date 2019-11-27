
from sklearn import tree
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


classifier = tree.DecisionTreeClassifier() 

classifier = classifier.fit(features, labels) 

print (classifier.predict([[356,9.2,9.2]]))

