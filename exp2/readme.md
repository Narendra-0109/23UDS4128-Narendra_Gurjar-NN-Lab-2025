## Objective

This project implements the **Perceptron Learning Algorithm** using **NumPy** in Python. The goal is to evaluate the performance of a single perceptron on the **NAND** and **XOR** truth tables and then extend it to a **Multi-Layer Perceptron (MLP)** for classifying Boolean functions.

## Description of the Model

This implementation follows a **multi-layer perceptron (MLP)** approach where:

- Individual perceptrons are trained to recognize Boolean functions (e.g., NAND, custom functions).
- The outputs of these perceptrons are then combined to train a final perceptron for binary classification.
- The model is evaluated using **accuracy**.

## Python Implementation

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias column
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]

    def evaluate(self, X, y):
        y_pred = np.array([self.predict(x) for x in X])
        accuracy = np.mean(y_pred == y) * 100
        return accuracy, y_pred

def train_perceptron(X, y, name):
    p = Perceptron(input_size=X.shape[1])
    p.train(X, y)
    accuracy, predictions = p.evaluate(X, y)
    print(f"{name} Accuracy: {accuracy:.2f}% | Predictions: {predictions}")
    return predictions, y

fun_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

fun1_y = np.array([0, 0, 0, 1])  
fun2_y = np.array([0, 0, 1, 0])  
fun3_y = np.array([0, 1, 0, 0])  
fun4_y = np.array([1, 0, 0, 0]) 

fun1_predictions, _ = train_perceptron(fun_X, fun1_y, "Fun1")
fun2_predictions, _ = train_perceptron(fun_X, fun2_y, "Fun2")
fun3_predictions, _ = train_perceptron(fun_X, fun3_y, "Fun3")
fun4_predictions, _ = train_perceptron(fun_X, fun4_y, "Fun4")

final_X = np.column_stack([fun1_predictions, fun2_predictions, fun3_predictions, fun4_predictions])
final_y = np.array([0, 1, 1, 0])

final_predictions, actual_y = train_perceptron(final_X, final_y, "Final Perceptron")
```
## Explanation of the Code

### Libraries Used
- **NumPy**: Used for numerical operations and matrix calculations.

### Perceptron Class
The **Perceptron** class models the perceptron algorithm, which initializes random weights and applies a learning rule based on error correction. The key components are:
- **Weights**: Initially set to random values, including an additional bias term.
- **Activation Function**: A step function used to classify inputs, where a value greater than or equal to zero is classified as 1, and below zero is classified as 0.
- **Training Process**: The model learns by adjusting weights during each iteration based on the classification errors. This is done using the perceptron learning rule.

### Activation Function
The activation function used is a **threshold-based step function**, which classifies inputs as either `0` or `1` based on whether the weighted sum of the inputs is greater than or equal to zero.

### Training Process
- Implements the **Perceptron Learning Algorithm**: During training, weights are updated in the direction of the error (difference between predicted and actual output).
- The model iterates through the data for a predefined number of **epochs**, applying weight updates after each sample.

### Evaluation
- **Accuracy** is computed as the percentage of correct classifications out of all predictions made. This gives an overall performance metric for the perceptron model.

## Results and Performance

The model was trained and evaluated on several Boolean functions (NAND, XOR, etc.), and here are the results:

- **Fun1** corresponds to the NAND function.
- **Fun2** corresponds to another Boolean function.
- **Fun3** and **Fun4** represent other Boolean functions.
- The **Final Perceptron** is trained on the outputs from the previous perceptrons and achieves 100% accuracy in classifying the combined output.

These results show that the perceptron and multi-layer perceptron approach effectively learn and classify the Boolean functions with **100% accuracy**.

