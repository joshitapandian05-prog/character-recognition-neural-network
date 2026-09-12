"""
Image Quality Inspection for Character Recognition Dataset.

Checks the processed character images for potentially low-quality
samples based on average pixel intensity and pixel variation.
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\AIML_Project\Character_Recognition_NN\data\processed_dataset.npz"
OUTPUT_PATH = r"C:\AIML_Project\Character_Recognition_NN\results\image_quality_samples.png"


def inspect_image_quality():
    """
    Inspect processed images and visualize samples with unusual
    pixel statistics.

    The inspection is used for qualitative data-quality analysis.
    It does not automatically remove samples from the dataset.
    """

    data = np.load(DATA_PATH)

    X = data["X"]
    y = data["y"]

    # Calculate simple image-quality indicators.
    mean_intensity = np.mean(X, axis=1)
    pixel_std = np.std(X, axis=1)

    # Find samples with unusually low pixel variation.
    low_variation_indices = np.argsort(pixel_std)[:5]

    # Find samples with unusually high pixel variation.
    high_variation_indices = np.argsort(pixel_std)[-5:]

    selected_indices = np.concatenate(
        [low_variation_indices, high_variation_indices]
    )

    plt.figure(figsize=(12, 5))

    for position, index in enumerate(selected_indices):
        plt.subplot(2, 5, position + 1)

        image = X[index].reshape(28, 28)

        plt.imshow(image, cmap="gray")
        plt.title(
            f"Class {y[index]}\nStd: {pixel_std[index]:.3f}",
            fontsize=8
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print("=" * 60)
    print("IMAGE QUALITY INSPECTION COMPLETE")
    print("=" * 60)
    print("Total images inspected:", len(X))
    print("Images visualized:", len(selected_indices))
    print("Output saved to:")
    print(OUTPUT_PATH)
    print()
    print("Note: No images were removed automatically.")
    print("The inspection is for data-quality analysis only.")
    print("=" * 60)


if __name__ == "__main__":
    inspect_image_quality()