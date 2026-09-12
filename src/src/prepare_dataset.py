"""
Prepare a balanced 35-class character recognition dataset.

The dataset contains:
1-9  -> digits
10-35 -> uppercase letters A-Z

Class 0 (digit 0) is excluded because the assignment requires
digits 1-9 and uppercase letters A-Z.
"""

import csv
import numpy as np

CSV_PATH = r"C:\Users\JOSHITA\OneDrive\Desktop\digit_char_dataset.csv"
OUTPUT_PATH = r"data\processed_dataset.npz"

SAMPLES_PER_CLASS = 100
NUM_CLASSES = 35
IMAGE_SIZE = 784


def prepare_dataset(csv_path, output_path):
    """
    Extract a balanced subset of the required character classes,
    normalize pixel values, shuffle the samples, and save the
    processed dataset.

    The function retains classes 1-35, excludes digit 0, and
    selects 100 samples per class for a balanced experiment.

    Parameters
    ----------
    csv_path : str
        Path to the original character dataset CSV file.

    output_path : str
        Path where the processed NumPy dataset will be saved.
    """
    required_classes = set(range(1, 36))

    images = []
    labels = []

    class_counts = {class_id: 0 for class_id in required_classes}

    print("Preparing dataset...")
    print(f"Target samples per class: {SAMPLES_PER_CLASS}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        header = next(reader)
        class_index = len(header) - 1

        for row in reader:

            if not row:
                continue

            class_id = int(float(row[class_index]))

            # Ignore class 0
            if class_id not in required_classes:
                continue

            # Stop collecting after 100 samples for this class
            if class_counts[class_id] >= SAMPLES_PER_CLASS:
                continue

            pixels = np.array(
                row[:IMAGE_SIZE],
                dtype=np.float32
            )

            # Normalize pixel values from 0-255 to 0-1
            pixels = pixels / 255.0

            images.append(pixels)
            labels.append(class_id - 1)

            class_counts[class_id] += 1

            # Stop when all 35 classes have enough samples
            if all(
                count >= SAMPLES_PER_CLASS
                for count in class_counts.values()
            ):
                break

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Shuffle the dataset
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # Save processed dataset
    np.savez_compressed(
        output_path,
        X=X,
        y=y
    )

    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)

    print("Total images:", len(X))
    print("Image features:", X.shape[1])
    print("Number of classes:", len(np.unique(y)))

    print("\nClass distribution:")

    for class_id in range(NUM_CLASSES):
        print(
            f"Class {class_id}: "
            f"{np.sum(y == class_id)} samples"
        )

    print("\nSaved to:", output_path)
    print("=" * 50)


if __name__ == "__main__":
    prepare_dataset(CSV_PATH, OUTPUT_PATH)