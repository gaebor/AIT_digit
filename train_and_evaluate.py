import numpy
from collections import defaultdict
import sys
from classifier import make_classifier
from put_image import put_image
from preprocessor import preprocess

train_filename = "data/train.csv"
k=3
train_threshold = 20000
verbose=False

i=1
while i < len(sys.argv):
	if sys.argv[i] == "-k":
		k=int(sys.argv[i+1])
		i += 1
	elif sys.argv[i] == "-t":
		train_threshold=int(sys.argv[i+1])
		i += 1
	elif sys.argv[i] == "-v":
		verbose = True
	elif sys.argv[i] == "--help":
		print >>sys.stderr, "simple demo training and evaluation"
		print >>sys.stderr, "USAGE:", sys.argv[0], "[-t int] [-k int] [-v] [filename]"
		print >>sys.stderr, "\tfilename\ttraining data in official format:"
		print >>sys.stderr, "\t\thttps://www.kaggle.com/c/digit-recognizer"
		print >>sys.stderr, "\t\tdefault is '" + train_filename + "'"
		print >>sys.stderr, "\t\tuse '-' for stdin"
		print >>sys.stderr, "\t-t int\tnumber of training instances, the rest is for evaluation, default training is", train_threshold
		print >>sys.stderr, "\t-k int\tthe k parameter of KNN, default=", k
		print >>sys.stderr, "\t-v\tsets verbosity"
		print >>sys.stderr, "the predictions are written to the stdout"
		print >>sys.stderr, "informations are written to the stderr"
		exit()
	else:
		train_filename = sys.argv[i]
	i += 1

print >>sys.stderr, "reading labelled dataset from '" + train_filename + "'..."

input = open(train_filename, "r") if train_filename != "-" else sys.stdin

input.readline()

X = numpy.loadtxt(input, delimiter=",", dtype=numpy.uint8)

labels = X[:,0]
X=X[:,1:].astype(float)

print >>sys.stderr, "training KNN with", min(train_threshold, X.shape[0]), "training instances and k=", k, "..."
clf = make_classifier(preprocess(X[:train_threshold]), labels[:train_threshold], name="KNN", params=[k])

print >>sys.stderr, "making predicitions for", max(0,X.shape[0]-train_threshold), "instances ..."
predictions = clf.predict(preprocess(X[train_threshold:]))

print >>sys.stderr, "evaluating ..."

if verbose:
	for i in range(len(predictions)):
		print labels[train_threshold:][i], predictions[i]
		if labels[train_threshold:][i] != predictions[i]:
			print >>sys.stderr, "should be:", labels[train_threshold:][i], ", was:", predictions[i]
			put_image(X[train_threshold:][i], 0, sys.stderr)
			print >>sys.stderr
else:
	for i in range(len(predictions)):
		print labels[train_threshold:][i], predictions[i]

score = float(numpy.sum(predictions == labels[train_threshold:]))/len(predictions)
print >>sys.stderr, score