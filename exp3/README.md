MNIST Classification using TensorFlow (Without Keras)

**Objective:**
This project implements a three-layer neural network using TensorFlow (without Keras) to classify the MNIST handwritten digits dataset. The implementation demonstrates both feed-forward and back-propagation approaches.

 **Description of the Model:**
This is a three-layer neural network designed for classifying handwritten digits from the MNIST dataset.

 **Model Architecture:**
- **Input Layer (784 neurons):** Accepts flattened 28x28 pixel images.
- **Hidden Layer 1 (128 neurons):** Uses ReLU activation to learn non-linear features.
- **Hidden Layer 2 (64 neurons):** Uses ReLU activation for deeper representation.
- **Output Layer (10 neurons):** Uses softmax activation to classify digits (0-9).
- **Loss Function:** Categorical cross-entropy.
- **Optimizer:** Adam Optimizer.
- **Training:** Mini-batch Gradient Descent with batch size = 32.

 **Python Implementation**
```python
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

# Disable eager execution to use tf.compat.v1.placeholder
tf.compat.v1.disable_eager_execution()

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape input data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)  # Flattening 28x28 images
x_test = x_test.reshape(-1, 784)

# One-hot encoding for labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Reset default graph
ops.reset_default_graph()

# Define placeholders
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Define model parameters
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1))

W1 = init_weights([784, 128])
b1 = tf.Variable(tf.zeros([128]))

W2 = init_weights([128, 64])
b2 = tf.Variable(tf.zeros([64]))

W3 = init_weights([64, 10])
b3 = tf.Variable(tf.zeros([10]))

# Feed-forward pass
def forward_propagation(X):
    z1 = tf.matmul(X, W1) + b1
    a1 = tf.nn.relu(z1)

    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.relu(z2)

    z3 = tf.matmul(a2, W3) + b3
    output = tf.nn.softmax(z3)

    return output

y_pred = forward_propagation(X)

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))

# Back-propagation using Adam Optimizer
learning_rate = 0.01
optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

# Accuracy metric
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training
epochs = 20
batch_size = 32

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X: x_train, Y: y_train})
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Evaluate on test set
    test_acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print(f"\nTest Accuracy: {test_acc:.4f}")
```

**Code Explanation:**
**1. Load & Preprocess Data**
- Normalize images (x_train & x_test in range [0,1]).
- Flatten images (28x28 → 784).
- Convert labels to one-hot encoding.

**2. Initialize Model Parameters**
- Weights (W1, W2, W3) initialized with small random values.
- Biases (b1, b2, b3) initialized as zeros.

 **3. Feed-Forward Propagation**
- Layer 1: `a1 = ReLU(X * W1 + b1)`.
- Layer 2: `a2 = ReLU(a1 * W2 + b2)`.
- Output Layer: `softmax(a2 * W3 + b3)`.

 **4. Loss Calculation**
- Uses `softmax_cross_entropy_with_logits()`.

 **5. Backpropagation & Optimization**
- Uses `AdamOptimizer()`.
- Updates weights and biases.

**6. Training (Mini-Batch Gradient Descent)**
- Iterates through 20 epochs with batch size = 32.
- Prints training loss & accuracy.

 **7. Testing**
- Evaluates the trained model on test data.

 **My Comments:**
- ✅ **Good Implementation:** Uses raw TensorFlow without Keras for full control.
- ✅ **Efficient Training:** Uses mini-batch gradient descent.
- ✅ **Basic Model:** Works well for MNIST but may not generalize to complex tasks.
- 🔹 **Improvement Suggestion:** Use Adam optimizer instead of SGD for faster convergence.
- 🔹 **TensorFlow v2 Compatibility:** Use `tf.compat.v1.disable_eager_execution()` for TF2 users.


