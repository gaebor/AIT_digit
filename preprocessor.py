import numpy

def preprocess(features, name="None", params=[]):
	if name == "MyCleverPreprocessor":
		return features.astype(numpy.float)
	return features