"""
Train the 35-class character recognition neural network.

The neural network is implemented from scratch using NumPy.
Training includes mini-batch gradient descent and validation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

DATA_PATH = r"C:\AIML_Project\Character_Recognition_NN\data\split_dataset.npz"

MODEL_PATH = r"C:\AIML_Project\Character_Recognition_NN\data\trained_model.npz"

HISTORY_PATH = r"C:\AIML_Project\Character_Recognition_NN\data\training_history.npz"

RESULTS_DIR = r"C:\AIML_Project\Character_Recognition_NN\results"


# ---------------------------------------------------------
# TRAINING SETTINGS
# ---------------------------------------------------------

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.01


# ---------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------
def train_model():
    """
    Train the character recognition neural network.

    The function loads the prepared training and validation datasets,
    performs mini-batch gradient descent with backpropagation,
    records training and validation metrics, and saves the trained
    model and training history.
    """
    # Make sure required folders exist
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # Load dataset
    print("Loading dataset...")

    data = np.load(DATA_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

    print("\n" + "=" * 60)
    print("CHARACTER RECOGNITION NEURAL NETWORK")
    print("=" * 60)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_validation))
    print("Input features:", X_train.shape[1])
    print("Output classes:", 35)

    # -----------------------------------------------------
    # CREATE MODEL
    # -----------------------------------------------------

    model = NeuralNetwork(
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=35,
        learning_rate=LEARNING_RATE
    )

    # Reproducible shuffling
    rng = np.random.default_rng(42)

    # Store training history
    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []

    print("\nStarting training...")
    print("-" * 60)

    # -----------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------

    for epoch in range(EPOCHS):

        # Shuffle training data
        indices = rng.permutation(len(X_train))

        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0.0
        batches = 0

        # Mini-batch training
        for start in range(
            0,
            len(X_train),
            BATCH_SIZE
        ):

            end = start + BATCH_SIZE

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward propagation
            predictions = model.forward(X_batch)

            # Cross-entropy loss
            loss = model.cross_entropy_loss(
                y_batch,
                predictions
            )

            # Backpropagation + gradient descent
            model.backward(
                X_batch,
                y_batch
            )

            epoch_loss += loss
            batches += 1

        # Average training loss
        epoch_loss /= batches

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        validation_predictions = model.forward(
            X_validation
        )

        validation_loss = model.cross_entropy_loss(
            y_validation,
            validation_predictions
        )

        # Training accuracy
        training_predictions = model.predict(
            X_train
        )

        training_accuracy = np.mean(
            training_predictions == y_train
        )

        # Validation accuracy
        validation_accuracy = np.mean(
            np.argmax(
                validation_predictions,
                axis=1
            ) == y_validation
        )

        # Store history
        train_losses.append(epoch_loss)
        validation_losses.append(validation_loss)

        train_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        # Display progress
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Acc: {training_accuracy * 100:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy * 100:.2f}%"
        )

    # -----------------------------------------------------
    # SAVE TRAINING HISTORY
    # -----------------------------------------------------

    np.savez(
        HISTORY_PATH,
        train_losses=np.array(train_losses),
        validation_losses=np.array(validation_losses),
        train_accuracies=np.array(train_accuracies),
        validation_accuracies=np.array(
            validation_accuracies
        )
    )

    # -----------------------------------------------------
    # SAVE TRAINED MODEL
    # -----------------------------------------------------

    np.savez(
        MODEL_PATH,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        W3=model.W3,
        b3=model.b3
    )

    # -----------------------------------------------------
    # VERIFY MODEL FILE
    # -----------------------------------------------------

    model_exists = os.path.exists(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("Model saved to:")
    print(MODEL_PATH)

    print("\nFile exists:", model_exists)

    print("\nTraining history saved to:")
    print(HISTORY_PATH)

    # -----------------------------------------------------
    # TRAINING CURVES
    # -----------------------------------------------------

    epochs = range(1, EPOCHS + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        validation_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        validation_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.tight_layout()

    accuracy_path = os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=200
    )

    plt.close()

    print("\nTraining curves saved:")
    print(loss_path)
    print(accuracy_path)

    print("=" * 60)


# ---------------------------------------------------------
# PROGRAM START
# ---------------------------------------------------------

if __name__ == "__main__":
    train_model()