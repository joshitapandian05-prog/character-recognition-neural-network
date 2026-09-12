"""
Evaluate the trained character recognition neural network
on the unseen test dataset.
"""

import numpy as np
from neural_network import NeuralNetwork


DATA_PATH = r"C:\AIML_Project\Character_Recognition_NN\data\split_dataset.npz"

MODEL_PATH = r"C:\AIML_Project\Character_Recognition_NN\data\trained_model.npz"

RESULT_PATH = r"C:\AIML_Project\Character_Recognition_NN\results\test_predictions.npz"


CLASS_NAMES = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]


def load_model():
    """
    Load the trained neural network architecture and saved weights.

    Returns
    -------
    NeuralNetwork
        Neural network containing the trained weights and biases.
    """
    model = NeuralNetwork(
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=35,
        learning_rate=0.01
    )

    weights = np.load(MODEL_PATH)

    model.W1 = weights["W1"]
    model.b1 = weights["b1"]

    model.W2 = weights["W2"]
    model.b2 = weights["b2"]

    model.W3 = weights["W3"]
    model.b3 = weights["b3"]

    return model


def evaluate():
    """
    Evaluate the trained model using the unseen test dataset.

    The function calculates test accuracy, identifies incorrect
    predictions, displays prediction confidence, and saves the
    predictions and probabilities for further analysis.
    """
    data = np.load(DATA_PATH)

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("\nLoading trained model...")

    model = load_model()

    print("Model loaded successfully.")

    # Forward propagation
    probabilities = model.forward(X_test)

    # Select class with highest probability
    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # Calculate accuracy
    accuracy = np.mean(
        predictions == y_test
    )

    # Find incorrect predictions
    incorrect = np.where(
        predictions != y_test
    )[0]

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    print("Test samples:", len(X_test))

    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        "Correct predictions:",
        len(X_test) - len(incorrect)
    )

    print(
        "Incorrect predictions:",
        len(incorrect)
    )

    print("\nFirst 10 incorrect predictions:")
    print("-" * 60)

    for index in incorrect[:10]:

        actual = CLASS_NAMES[y_test[index]]

        predicted = CLASS_NAMES[
            predictions[index]
        ]

        confidence = probabilities[
            index,
            predictions[index]
        ]

        print(
            f"Sample {index}: "
            f"Actual = {actual}, "
            f"Predicted = {predicted}, "
            f"Confidence = {confidence * 100:.2f}%"
        )

    # Save predictions for later analysis
    np.savez(
        RESULT_PATH,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities
    )

    print("\nPredictions saved to:")
    print(RESULT_PATH)

    print("=" * 60)


if __name__ == "__main__":
    evaluate()