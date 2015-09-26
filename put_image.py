import numpy
import sys

def put_image(image, threshold=0.0, output=sys.stdout):
	if type(image) == type([]):
		A=numpy.array(image).reshape((28,28)) > threshold
	elif type(image) == type(numpy.zeros((1,1))):
		A=image.reshape((28,28)) > threshold
	else:
		print >>sys.stderr, "unknown type", type(image)
		return
	for i in range(28):
		for j in range(28):
			output.write('X' if A[i,j] else ' ')
		print >>output
