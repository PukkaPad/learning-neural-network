# coding: utf-8

import numpy as np

# X = hours sleeping, hours studying
# y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # max of X
y = y/100 # max score is 100

class Neural_Network(object):
	def __init__(self):
		#parameters
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenSize = 3

		# weights
		self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # 3X2
		self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # 3X1
	
	def forward(self, X):
		"""Forward propagation function. 
		It takes a dot product of the inputs (hours sleeping and hours studying) and weights (W1). Then , to ge the value of the hidden layer, it applies an activation function (sigmoid). Then it takes another dot product of the hidden layer and the 2nd set of weights (W2) and applies a final activation function (sigmoid) to receive an output.

		Args:
			X (object): numpy matrix

		Returns:
			y
		"""

		self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of 3x2 weights
		self.z2 = self.sigmoid(self.z) # activaton function
		self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2)  and 2nd set of 3X1 weight
		o = self.sigmoid(self.z3)  # final activation function

		return o

	def sigmoid(self, s):
		"""Sigmoid function.
		This is the activation function chosen for this example.

		Args:
			s (int): dot product

		Returns:
			sigmoid funtion of the dot product
		"""

		return 1/(1+np.exp(-s))


NN = Neural_Network()

# output
o = NN.forward(X)

print( "Predicted Output: \n" + str(o))
print( "Actual Output: \n" + str(y))