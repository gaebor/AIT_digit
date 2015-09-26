import numpy
import sys
from classifier import make_classifier
from preprocessor import preprocess

train_filename = "data/train.csv"
test_filename = "data/test.csv"

k=3
train_threshold = 20000

i=1
while i < len(sys.argv):
	if sys.argv[i] == "-k":
		k=int(sys.argv[i+1])
		i += 1
	elif sys.argv[i] == "-test":
		test_filename=sys.argv[i+1]
		i += 1
	elif sys.argv[i] == "-train":
		train_filename=sys.argv[i+1]
		i += 1
	elif sys.argv[i] == "-t":
		train_threshold=int(sys.argv[i+1])
		i += 1
	elif sys.argv[i] == "--help":
		print >>sys.stderr, "simple demo training and evaluation"
		print >>sys.stderr, "USAGE:", sys.argv[0], "[-t int] [-k int] [-test filename] [-train filename]"
		print >>sys.stderr, "\t-train filename\ttraining data in official format:"
		print >>sys.stderr, "\t\thttps://www.kaggle.com/c/digit-recognizer/data"
		print >>sys.stderr, "\t\tdefault is '" + train_filename + "'"
		print >>sys.stderr, "\t\tuse '-' for stdin"
		print >>sys.stderr, "\t-test filename\ttest data in official format"
		print >>sys.stderr, "\t-t int\tnumber of training instances, default is", train_threshold
		print >>sys.stderr, "\t-k int\tthe k parameter of KNN, default=", k
		print >>sys.stderr, "the predictions are written to the stdout"
		print >>sys.stderr, "informations are written to the stderr"
		exit()
	else:
		train_filename = sys.argv[i]
	i += 1

print >>sys.stderr, "reading train dataset from '" + train_filename + "'..."

input = open(train_filename, "r") if train_filename != "-" else sys.stdin

input.readline()

X = numpy.loadtxt(input, delimiter=",", dtype=numpy.uint8)
input.close()

labels = X[:,0]
X=X[:,1:].astype(float)

print >>sys.stderr, "training KNN with", min(train_threshold, X.shape[0]), "training instances and k=", k, "..."

clf = make_classifier(preprocess(X[:train_threshold]), labels[:train_threshold], name="KNN", params=[k])

print >>sys.stderr, "reading test dataset from '" + test_filename + "'..."

input = open(test_filename, "r")
input.readline()

Y= numpy.loadtxt(input, delimiter=",", dtype=numpy.uint8)

print >>sys.stderr, "making predictions ..."
predictions = clf.predict(preprocess(Y))

for i in range(len(predictions)):
	print predictions[i]
