
from sklearn import tree

features = [[204,7.5,9.2], [362,9.6,9.2], [162,7.5,7.1], [162,7.4,7.2]]

labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier() 

classifier = classifier.fit(features, labels) 

print (classifier.predict([[356,9.2,9.2]]))

