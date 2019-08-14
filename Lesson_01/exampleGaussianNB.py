import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # Features
Y = np.array([1, 1, 1, 2, 2, 2]) # Labels, either class 1 or class 2

clf = GaussianNB()
clf.fit(X, Y) # This is where you give the training data, and the classification algorithm learns
print(clf.predict([[-0.8, -1]])) # Predict what the label is for this point