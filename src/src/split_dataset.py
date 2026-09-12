"""
Split the processed character dataset into
training, validation, and testing sets.
"""

import numpy as np

INPUT_PATH = r"data\processed_dataset.npz"
OUTPUT_PATH = r"data\split_dataset.npz"


def split_dataset(input_path, output_path):
    """
    Split the processed dataset into training, validation, and test sets.

    The split maintains representation from all 35 character classes
    and uses a fixed random seed to make the experiment reproducible.

    Parameters
    ----------
    input_path : str
        Path to the processed dataset containing images and labels.

    output_path : str
        Path where the train, validation, and test datasets will be saved.
    """
    data = np.load(input_path)

    X = data["X"]
    y = data["y"]

    rng = np.random.default_rng(42)

    train_indices = []
    validation_indices = []
    test_indices = []

    # Stratified split: maintain all 35 classes
    for class_id in range(35):

        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)

        n = len(class_indices)

        train_end = int(0.70 * n)
        validation_end = int(0.85 * n)

        train_indices.extend(class_indices[:train_end])
        validation_indices.extend(
            class_indices[train_end:validation_end]
        )
        test_indices.extend(
            class_indices[validation_end:]
        )

    # Shuffle each split
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_validation = X[validation_indices]
    y_validation = y[validation_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test
    )

    print("\n" + "=" * 50)
    print("DATASET SPLIT COMPLETE")
    print("=" * 50)

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Testing samples:    {len(X_test)}")

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_validation:", X_validation.shape)
    print("X_test:", X_test.shape)

    print("\nNumber of classes:", len(np.unique(y)))

    print("\nSaved to:", output_path)
    print("=" * 50)


if __name__ == "__main__":
    split_dataset(INPUT_PATH, OUTPUT_PATH)