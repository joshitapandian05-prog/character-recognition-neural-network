"""
Generate and save the confusion matrix for the test predictions.

The confusion matrix provides a class-by-class view of the model's
classification performance across all 35 character classes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


RESULT_PATH = r"C:\AIML_Project\Character_Recognition_NN\results\test_predictions.npz"

OUTPUT_PATH = r"C:\AIML_Project\Character_Recognition_NN\results\confusion_matrix.png"


CLASS_NAMES = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]


def create_confusion_matrix(y_true, y_pred):
    """
    Create a 35-class confusion matrix using NumPy.

    Parameters
    ----------
    y_true : numpy.ndarray
        Actual class labels.

    y_pred : numpy.ndarray
        Predicted class labels.

    Returns
    -------
    numpy.ndarray
        Matrix containing the number of predictions for each
        actual and predicted class pair.
    """
    num_classes = len(CLASS_NAMES)

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1

    return matrix


def save_confusion_matrix():
    """
    Documentation goes here.
    More documentation goes here.
    """

    # Actual code starts here
    data = np.load(RESULT_PATH)
    y_test = data["y_test"]
    predictions = data["predictions"]

    matrix = create_confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(matrix)

    plt.figure(figsize=(14, 12))

    plt.imshow(matrix)

    plt.title(
        "Confusion Matrix - 35 Class Character Recognition"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(35),
        CLASS_NAMES,
        rotation=90
    )

    plt.yticks(
        range(35),
        CLASS_NAMES
    )

    plt.colorbar()

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    plt.close()

    print("\nConfusion matrix saved to:")
    print(OUTPUT_PATH)

    print("=" * 60)


if __name__ == "__main__":
    save_confusion_matrix()