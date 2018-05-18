# coding: utf-8

import numpy as np

# X = hours sleeping, hours studying
# y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4,8]), dtype=float)

# scale units
X_scaled = X/np.amax(X, axis=0)  # max of X
y_scaled = y/100  # max score is 100
xPredicted_scaled = xPredicted/np.amax(xPredicted, axis=0)  # maximum of xPredicted (the input data for the prediction)


class Neural_Network(object):

    def __init__(self):
        # parameters
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

        self.z = np.dot(X_scaled, self.W1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activaton function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and 2nd set of 3X1 weight
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

    def sigmoidPrime(self, s):
        """ Derivatide of the sigmoid.
        This funtion gives the rate of change (slope) of the activation function at output sum.

        Args:
            s (int): dot product

        Returns:
            rate of change of the activation function
        """
        return s*(1 - s)

    def backward(self, X, y, o):
        """ Backward propagation function.
        This function uses a loss function to calcuate how far the network was from th target input.
        The loss function is represented by using the mean sum squared loss:
        Loss = sum((0.5) * (predicted output - y)^2), where y is the actual output.
        The objective is to get a loss as close to zero as possible. For that, it is necessary to alter the weights. So the derivative of the loss function is used to understand how the weights affect the input. The method is called gradient descent.

        The four steps applied in this function that will calculate the incremental change to the weights are:

        1) Find the margin of error of the output layer (o) by taking the difference of the predicted output and the actual output (y)

        2) Apply the derivative of the sigmoid activation function to the output layer error. This result will be called the delta output sum.

        3) Use the delta output sum of the output layer error to figure out how much the z2 (hidden) layer contributed to the output error by performing a dot product with the second weight matrix. This will be called the z2 error.

        4) Calculate the delta output sum for the z2 layer by applying the derivative of the sigmoid activation function (just like step 2).

        Args:
            X (object): numpy matrix
            y (object): actual output
            o (object): predicted output

        """
        self.o_error = y - o  # error in input
        self.o_delta = self.o_error*self.sigmoidPrime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much the hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train(self, X, y):
        """Train function.
        This function initiates the forward and backward functions.

        Args:
            X (object): numpy matrix
            y (object): actual output
        """

        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        """This function the predicted output for xPredicted
        """
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted_scaled))
        print("Output (scaled): \n" + str(self.forward(xPredicted_scaled)))
        print("Output (not-scaled): \n" + str(y*np.amax(self.forward(xPredicted_scaled), axis=0)))

    def saveWeights(self):
        """Function saves the trained weights
        """
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")


NN = Neural_Network()

for i in range(1000000):  # trains the NN 1,000 times
    print("Input (scaled): \n" + str(X_scaled))
    print("Input (not-scaled) : \n" + str(X*np.amax(X_scaled, axis=0)))
    #X = X/np.amax(X, axis=0)  # max of X

    print("Actual (scaled) Output: \n" + str(y_scaled))
    print("Predicted (scaled) Output: \n" + str(NN.forward(X_scaled)))
    print("Actual (not-scaled) Output: \n" + str(y*np.amax(y_scaled, axis=0)))
    print("Predicted (not-scaled) Output: \n" + str(y*np.amax(NN.forward(X_scaled), axis=0)))
    print("Loss: \n" + str(np.mean(np.square(y_scaled - NN.forward(X_scaled)))))  # mean sum squared loss
    print("\n")
    NN.train(X_scaled, y_scaled)

    NN.saveWeights()
    NN.predict()