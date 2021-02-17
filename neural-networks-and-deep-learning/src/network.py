"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """Neural Network

        The biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.

        :param list size: 
            first index is number of input layer neurons.
            last index in number of output layer neurons.
            each index in between first and last is a hidden layer of neurons.

        Example:
            [2, 3, 1]     # 3-layer network, 2 input, 3 hidden, 1 output
            [2, 3, 4, 1]  # 4-layer network, 2 input, 3 hidden, 4 hidden, 1 output
        """
        self.num_layers = len(sizes)  # e.g. 3
        self.sizes = sizes            # e.g. [2, 3, 1]

        """
        # each neuron in each layer which is not input layer gets a bias assigned to it
        # e.g. for [2, 3, 1] it will only create random biases for [3, 1]
        [
            [
                [ 0.11990344]
                [-1.38834335]
                [-0.50051942]
            ],
            [
                [-0.06048575]
            ] 
        ]
        """
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 

        """
        # weights are the lines between each neuron
        # e.g. weights[1] is a numpy matrix storing the weights connecting the second and third layers
        # each input layer neuron goes to each second layer neuron  2 * 3 = 6 total weights
        # each second layer neuron goes to each output layer neuron 3 * 1 = 3 total weights
        [
            [
                [-1.58724458  1.39180206]
                [-0.81362494 -1.50619335]
                [-1.43461051 -3.25606635]
            ],
            [
                [-0.55330222  0.43328301  0.26116272]
            ]
        ]
        """
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # NOTE: this produces pairwise tuples
        # x, y in zip(sizes[:-1], sizes[1:])
        # e.g. [(2, 3), (3, 1)] for sizes=[2, 3, 1]
        # so that we can create a matrices between each (see above)


    def feedforward(self, a):
        """Applies sigmoid equation to weights and biases"""
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a


    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.  

        :param list training_data: list of (x, y) representing the training inputs and the desired outputs.
        :param int epochs: number of epochs to train for (iterations of training).
        :param int mini_batch_size: size of mini-batches to use when sampling.
        :param float eta: learning rate. (eta is greek symbol which looks like an 'n')
        :param list test_data: (optional) If provided then the network will be evaluated against the test data 
            after each epoch, and partial progress printed out.  This is useful for tracking progress, 
            but slows things down substantially.
        """
        len_training_data = len(training_data)
        if test_data is not None:
            len_test_data = len(test_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, len_training_data, mini_batch_size)]

            # for each mini_batch, apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len_test_data)
            else:
                print "Epoch {0} complete".format(j)


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
