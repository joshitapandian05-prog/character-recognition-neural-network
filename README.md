# Character Recognition Neural Network from Scratch

## 1. Project Overview

This project implements a 35-class handwritten character recognition neural network completely from scratch using NumPy.

The system recognizes:

- Digits 1-9
- Uppercase letters A-Z

The neural network was implemented without TensorFlow, PyTorch, or Keras.

---

## 2. Objective

The objective is to develop a neural network capable of classifying handwritten characters into 35 different classes using a NumPy-based implementation of the complete neural network training process.

The implementation includes forward propagation, activation functions, loss calculation, backpropagation, gradient descent, validation, testing, confusion matrix generation, and error analysis.

---

## 3. Dataset

The original dataset contains 36 classes numbered from 0 to 35.

Class 0 represents digit 0. Since the assignment requires digits 1-9 and uppercase letters A-Z, class 0 was excluded.

The remaining 35 classes were used.

A balanced subset of 100 samples was selected from each class.

**Total samples:** 3500

**Image resolution:** 28 x 28 pixels

**Input features:** 784

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Selected the required 35 classes.
2. Excluded digit 0.
3. Selected 100 samples from each class.
4. Converted pixel values into floating-point values.
5. Normalized pixel values from 0-255 to 0-1.
6. Represented each 28 x 28 image as 784 input features.
7. Randomly shuffled the dataset using a fixed random seed of 42.
8. Created training, validation, and test datasets.

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 2450 |
| Validation | 525 |
| Testing | 525 |
| **Total** | **3500** |

---

## 5. Neural Network Architecture

The implemented neural network is a fully connected feed-forward neural network.

```text
Input Layer
784 neurons
     |
     v
Hidden Layer 1
128 neurons
ReLU
     |
     v
Hidden Layer 2
64 neurons
ReLU
     |
     v
Output Layer
35 neurons
Softmax
     |
     v
Predicted Character
```

### Architecture Details

| Layer | Size | Activation |
|---|---:|---|
| Input | 784 | None |
| Hidden 1 | 128 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 35 | Softmax |

---

## 6. Algorithms Implemented

The neural network was implemented using NumPy.

### Forward Propagation

Input data is passed through the two hidden layers and finally through the output layer to produce class probabilities.

The computations are:

```text
Z1 = XW1 + b1
A1 = ReLU(Z1)

Z2 = A1W2 + b2
A2 = ReLU(Z2)

Z3 = A2W3 + b3
Output = Softmax(Z3)
```

### ReLU Activation

ReLU is used in the hidden layers to introduce non-linearity.

```text
ReLU(x) = max(0, x)
```

### Softmax

Softmax converts the output scores into probabilities for the 35 character classes.

The class with the highest probability is selected as the prediction.

### Cross-Entropy Loss

Cross-entropy loss measures the difference between the predicted probability distribution and the actual class.

### Backpropagation

Backpropagation calculates gradients by propagating the prediction error backward through the network.

### Gradient Descent

The calculated gradients are used to update the network weights and biases.

```text
New Weight = Old Weight - Learning Rate x Gradient
```

---

## 7. Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 30 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Random Seed | 42 |
| Optimizer | Mini-Batch Gradient Descent |

The model was trained using mini-batches and the validation dataset was evaluated after every epoch.

---

## 8. Results

The model was trained for 30 epochs and evaluated using training, validation, and unseen test datasets.

### Final Training Metrics

| Metric | Result |
|---|---:|
| Training Loss | 0.9783 |
| Validation Loss | 1.1311 |
| Training Accuracy | 76.24% |
| Validation Accuracy | 71.43% |

### Test Performance

| Metric | Result |
|---|---:|
| Test Samples | 525 |
| Correct Predictions | 363 |
| Incorrect Predictions | 162 |
| **Test Accuracy** | **69.14%** |

The final model achieved a test accuracy of **69.14%** on 525 unseen test samples.

The difference between training, validation, and test accuracy indicates a moderate generalization gap, showing that the model performs better on the training data than on unseen data.

---

## 9. Training and Validation Curves

Training and validation performance were recorded during the 30 training epochs.

The generated graphs are available in:

```text
results/loss_curve.png
results/accuracy_curve.png
```

The loss curve shows how the training and validation loss changes during training.

The accuracy curve shows the training and validation classification accuracy across the epochs.

These curves help evaluate the learning behavior and generalization of the model.

---

## 10. Confusion Matrix

A 35 x 35 confusion matrix was generated using the test predictions.

The confusion matrix is available at:

```text
results/confusion_matrix.png
```

The rows represent the actual character classes and the columns represent the predicted classes.

The confusion matrix helps identify correctly classified characters and character classes that are frequently confused with each other.

---

## 11. Error Analysis

Five incorrect predictions were selected from the test results for qualitative analysis.

| Case | Actual | Predicted | Possible Reason |
|---|---|---|---|
| 1 | E | P | Similar handwritten stroke structure |
| 2 | H | 4 | Similar vertical/intersecting stroke patterns |
| 3 | P | A | Variation in handwritten shape |
| 4 | C | O | Similar rounded visual structure |
| 5 | 5 | R | Similar curved stroke structure |

These are possible explanations based on the visual characteristics of the handwritten samples rather than confirmed causes.

Classification errors can occur because of:

- Different handwriting styles
- Stroke thickness
- Image quality
- Character shape variation
- Similar visual structures between different classes

The five selected incorrect prediction examples are available at:

```text
results/wrong_predictions.png
```

---

## 12. Project Workflow

```text
Dataset
   |
   v
Data Inspection
   |
   v
Class Selection
   |
   v
Balanced Sampling
   |
   v
Image Preprocessing
   |
   v
Pixel Normalization
   |
   v
Train / Validation / Test Split
   |
   v
Neural Network Initialization
   |
   v
Forward Propagation
   |
   v
Loss Calculation
   |
   v
Backpropagation
   |
   v
Gradient Descent
   |
   v
Validation
   |
   v
Test Evaluation
   |
   v
Confusion Matrix
   |
   v
Error Analysis
```

---

## 13. Project Structure

```text
Character_Recognition_NN
|
+-- README.md
|
+-- data
|   +-- processed_dataset.npz
|   +-- split_dataset.npz
|   +-- trained_model.npz
|   +-- training_history.npz
|
+-- results
|   +-- accuracy_curve.png
|   +-- loss_curve.png
|   +-- confusion_matrix.png
|   +-- test_predictions.npz
|   +-- wrong_predictions.png
|
+-- src
    +-- check_classes.py
    +-- confusion_matrix.py
    +-- evaluate.py
    +-- inspect_dataset.py
    +-- neural_network.py
    +-- prepare_dataset.py
    +-- split_dataset.py
    +-- train.py
    +-- wrong_predictions.py
```

---

## 14. Limitations

The experiment uses 100 samples per class to maintain a balanced dataset while keeping the training process computationally efficient.

The model achieved 69.14% test accuracy, indicating that there is room for further improvement.

Possible improvements include:

- Increasing the number of training samples.
- Training for more epochs.
- Tuning the learning rate.
- Increasing hidden-layer capacity.
- Applying data augmentation.
- Performing additional hyperparameter experiments.
- Exploring different neural network architectures.

---

## 15. Future Enhancements

Future versions of the system could include:

- Larger training datasets.
- Additional hidden layers.
- Hyperparameter optimization.
- Data augmentation.
- Improved optimization strategies.
- More extensive error analysis.
- A graphical interface for drawing or uploading characters.
- Real-time character prediction.

---

## 16. Conclusion

A 35-class handwritten character recognition neural network was successfully implemented from scratch using NumPy.

The complete pipeline includes dataset inspection, class selection, balanced sampling, preprocessing, normalization, train-validation-test splitting, neural network initialization, forward propagation, loss calculation, backpropagation, gradient descent, validation, testing, confusion matrix generation, and error analysis.

The trained model was evaluated on 525 unseen test samples and achieved a test accuracy of **69.14%**, with **363 correct predictions** and **162 incorrect predictions**.

The project demonstrates the fundamental working principles of a neural network without relying on high-level deep learning frameworks.
