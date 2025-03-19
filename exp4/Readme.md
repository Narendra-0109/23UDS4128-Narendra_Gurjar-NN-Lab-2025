# Three-Layer Neural Network for MNIST Classification

## Objective
This project evaluates the performance of a three-layer neural network by varying activation functions, hidden layer sizes, learning rates, batch sizes, and the number of epochs.

## Description
The model is implemented using TensorFlow (without Keras) to classify handwritten digits from the MNIST dataset. It explores the impact of different batch sizes and epoch configurations on performance.

## Model Architecture
- **Input Layer**: Flattened 28×28 images (784 features)
- **Hidden Layer 1**: 128 neurons, ReLU activation
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons (digits 0-9), softmax activation

## Training Details
- **Loss Function**: Categorical cross-entropy
- **Optimizer**: Adam
- **Learning Rate**: 0.01
- **Batch Sizes**: 100, 10, 1
- **Epochs**: 100, 50, 10
- **Training Method**: Mini-batch Gradient Descent

## Evaluation Metrics
- **Training Loss**
- **Training Accuracy**
- **Test Accuracy**
- **Confusion Matrix**

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Usage
```bash
pip install tensorflow numpy matplotlib
python train_model.py
```



# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Define accuracy
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            for epoch in range(epochs):
                for i in range(0, len(x_train), batch_size):
                    batch_x = x_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X: x_train, Y: y_train})
                test_acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
                print(f"Batch Size: {batch_size}, Epoch: {epoch+1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
```

## Results
The model performance is evaluated for different batch sizes and epochs, analyzing its impact on accuracy and loss. The confusion matrix provides insights into misclassified digits.
