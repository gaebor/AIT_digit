import numpy
from sklearn.neighbors import KNeighborsClassifier

def make_classifier(data, labels, name="KNN", params=[]):
	cls = None
	if name == "KNN":
		clf = KNeighborsClassifier(int(params[0]) if len(params)>0 else 3)
		clf.fit(data, labels)
	return clf