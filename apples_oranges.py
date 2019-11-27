
from sklearn import tree

features = [[192,8.4,7.3], [180,8.0,6.8], [342,9.0,9.4], [356,9.2,9.2]]

labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier() 

classifier = classifier.fit(features, labels) 

print (classifier.predict([[166,6.9,7.3]]))

