"""
Display and save five incorrect test predictions.
"""

import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]


def show_wrong_predictions():
    """Save the first five incorrectly classified images."""

    data = np.load(
        "results/test_predictions.npz"
    )

    X_test = data["X_test"]
    y_test = data["y_test"]
    predictions = data["predictions"]

    incorrect = np.where(
        predictions != y_test
    )[0]

    if len(incorrect) < 5:
        print("Fewer than five incorrect predictions.")
        return

    selected = incorrect[:5]

    plt.figure(figsize=(12, 7))

    for position, index in enumerate(selected):

        image = X_test[index].reshape(28, 28)

        plt.subplot(1, 5, position + 1)

        plt.imshow(
            image,
            cmap="gray"
        )

        plt.title(
            f"True: {CLASS_NAMES[y_test[index]]}\n"
            f"Pred: {CLASS_NAMES[predictions[index]]}"
        )

        plt.axis("off")

        print(
            f"Case {position + 1}: "
            f"True = {CLASS_NAMES[y_test[index]]}, "
            f"Predicted = {CLASS_NAMES[predictions[index]]}"
        )

    plt.tight_layout()

    plt.savefig(
        "results/wrong_predictions.png",
        dpi=200
    )

    plt.close()

    print(
        "\nSaved to: results/wrong_predictions.png"
    )


if __name__ == "__main__":
    show_wrong_predictions()