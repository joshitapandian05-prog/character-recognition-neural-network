"""
Neural Network for 35-Class Character Recognition.

This module implements a fully connected feed-forward neural network
from scratch using NumPy.

The network performs:
    - Forward propagation
    - ReLU activation
    - Softmax classification
    - Cross-entropy loss calculation
    - Backpropagation
    - Mini-batch gradient descent

The model accepts 784 input features representing a 28 x 28 grayscale
image and produces probabilities for 35 character classes.
"""

import numpy as np


class NeuralNetwork:
    """
    Fully connected neural network for handwritten character classification.

    The network contains two hidden layers followed by a Softmax output
    layer. ReLU activation is used in the hidden layers.

    Architecture:
        Input: 784 neurons
        Hidden Layer 1: 128 neurons + ReLU
        Hidden Layer 2: 64 neurons + ReLU
        Output Layer: 35 neurons + Softmax
    """

    def __init__(
        self,
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=35,
        learning_rate=0.01
    ):
        """
        Initialize the neural network architecture and parameters.

        Parameters
        ----------
        input_size : int
            Number of input features. The default value of 784
            corresponds to a flattened 28 x 28 image.

        hidden1_size : int
            Number of neurons in the first hidden layer.

        hidden2_size : int
            Number of neurons in the second hidden layer.

        output_size : int
            Number of output classes. The default is 35.

        learning_rate : float
            Step size used during gradient descent parameter updates.
        """

        self.learning_rate = learning_rate

        # He initialization for the ReLU-based hidden layers.
        self.W1 = (
            np.random.randn(input_size, hidden1_size)
            * np.sqrt(2.0 / input_size)
        )
        self.b1 = np.zeros((1, hidden1_size))

        self.W2 = (
            np.random.randn(hidden1_size, hidden2_size)
            * np.sqrt(2.0 / hidden1_size)
        )
        self.b2 = np.zeros((1, hidden2_size))

        # Output layer parameters.
        self.W3 = (
            np.random.randn(hidden2_size, output_size)
            * np.sqrt(2.0 / hidden2_size)
        )
        self.b3 = np.zeros((1, output_size))

    def relu(self, z):
        """
        Apply the Rectified Linear Unit (ReLU) activation function.

        ReLU introduces non-linearity by retaining positive values
        and replacing negative values with zero.

        Parameters
        ----------
        z : numpy.ndarray
            Input activation values.

        Returns
        -------
        numpy.ndarray
            Activated values after applying ReLU.
        """

        return np.maximum(0, z)

    def relu_derivative(self, z):
        """
        Calculate the derivative of the ReLU activation function.

        The derivative is 1 for positive input values and 0 for
        zero or negative input values.

        Parameters
        ----------
        z : numpy.ndarray
            Input values before applying ReLU.

        Returns
        -------
        numpy.ndarray
            Element-wise derivative values.
        """

        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Convert output scores into a probability distribution.

        A numerical-stability adjustment is applied by subtracting
        the maximum value in each row before calculating exponentials.

        Parameters
        ----------
        z : numpy.ndarray
            Output scores from the final linear layer.

        Returns
        -------
        numpy.ndarray
            Probability distribution over the 35 classes.
        """

        z = z - np.max(z, axis=1, keepdims=True)

        exp_values = np.exp(z)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

    def forward(self, X):
        """
        Perform forward propagation through the network.

        The input passes through two fully connected layers with
        ReLU activation and then through the Softmax output layer.

        Parameters
        ----------
        X : numpy.ndarray
            Input batch containing flattened image features.

        Returns
        -------
        numpy.ndarray
            Predicted probability distribution for each input sample.
        """

        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.output = self.softmax(self.z3)

        return self.output

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Calculate the multi-class cross-entropy loss.

        The loss is calculated using the predicted probability assigned
        to the correct class for each sample.

        Parameters
        ----------
        y_true : numpy.ndarray
            Array containing the true class indices.

        y_pred : numpy.ndarray
            Array containing predicted class probabilities.

        Returns
        -------
        float
            Mean cross-entropy loss across the batch.
        """

        batch_size = len(y_true)

        correct_probabilities = y_pred[
            np.arange(batch_size),
            y_true
        ]

        # Prevent log(0) and maintain numerical stability.
        correct_probabilities = np.clip(
            correct_probabilities,
            1e-12,
            1.0
        )

        loss = -np.mean(
            np.log(correct_probabilities)
        )

        return loss

    def backward(self, X, y_true):
        """
        Perform backpropagation and update network parameters.

        Gradients are calculated from the output layer back through
        both hidden layers. The resulting gradients are then used
        to update weights and biases using gradient descent.

        Parameters
        ----------
        X : numpy.ndarray
            Input batch used during the forward pass.

        y_true : numpy.ndarray
            True class indices corresponding to the input batch.
        """

        batch_size = X.shape[0]

        # Calculate gradient at the output layer.
        dz3 = self.output.copy()

        dz3[
            np.arange(batch_size),
            y_true
        ] -= 1

        dz3 /= batch_size

        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Propagate gradients through hidden layer 2.
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivative(self.z2)

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Propagate gradients through hidden layer 1.
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update parameters using gradient descent.
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def predict(self, X):
        """
        Predict the character class for each input sample.

        Parameters
        ----------
        X : numpy.ndarray
            Input samples represented as flattened image features.

        Returns
        -------
        numpy.ndarray
            Predicted class index for each input sample.
        """

        probabilities = self.forward(X)

        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y):
        """
        Calculate classification accuracy.

        Parameters
        ----------
        X : numpy.ndarray
            Input samples.

        y : numpy.ndarray
            True class labels.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """

        predictions = self.predict(X)

        return np.mean(predictions == y)