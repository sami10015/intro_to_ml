import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # Features
Y = np.array([1, 1, 1, 2, 2, 2]) # Labels, either class 1 or class 2

clf = GaussianNB()
clf.fit(X, Y) # This is where you give the training data, and the classification algorithm learns
prediction = clf.predict([[-0.8, -1]]) # Predict what the label is for this point
print(prediction)

# Here is a way that you can predict how accurate your classification is
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(sample_of_labels, prediction)