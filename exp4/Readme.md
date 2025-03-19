
Objective:
WAP to evaluate the performance of implemented three-layer neural network with variations in activation functions, size of hidden layer, learning rate, batch size and number of epochs.
Description of the Model:
This is a three-layer neural network implemented using TensorFlow (without Keras) for classifying handwritten digits from the MNIST dataset.The model consists of multiple layers and is trained using Mini-batch Gradient Descent with different batch sizes and epoch configurations.

It explores the effects of batch size (100, 10, 1) and epochs (100, 50, 10) on model performance.

🔹 Model Architecture:

Input: Flattened 28×28 images (784 features).
Hidden Layer 1: 128 neurons, ReLU activation.
Hidden Layer 2: 64 neurons, ReLU activation.
Output Layer: 10 neurons (digits 0-9), softmax activation.
Loss Function: Categorical cross-entropy.
Optimizer: Adam.
learning rate = 0.01.
Different batch sizes: 100,10,1.
Different epochs: 100,50,10.
Metrics: Train loss, train accuracy, test accuracy, confusion matrix.
Python Implementation

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape input data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# One-hot encoding for labels
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]

# Training configurations
batch_sizes = [100, 10, 1]
epochs_list = [100, 50, 10]
learning_rate = 0.01
results = {}

# Separate loop for each batch size
for batch_size in batch_sizes:
    print(f"\nTraining for Batch Size={batch_size}")

    # Separate loop for each epoch value
    for epochs in epochs_list:
        print(f"\nTraining for {epochs} Epochs")
        train_losses = []
        train_accuracies = []
        start_time = time.time()

        # New Model Initialization for Every Training
        tf.compat.v1.reset_default_graph()  # Clears previous graph

        # Define placeholders
        X = tf.compat.v1.placeholder(tf.float32, [None, 784])
        Y = tf.compat.v1.placeholder(tf.float32, [None, 10])

        # Initialize Weights & Biases inside the loop (New model every time)
        def init_weights(shape):
            return tf.Variable(tf.random.normal(shape, stddev=0.1))

        W1 = init_weights([784, 128])
        b1 = tf.Variable(tf.zeros([128]))
        W2 = init_weights([128, 64])
        b2 = tf.Variable(tf.zeros([64]))
        W3 = init_weights([64, 10])
        b3 = tf.Variable(tf.zeros([10]))

        # Forward propagation inside loop
        def forward_propagation(X):
            z1 = tf.matmul(X, W1) + b1
            a1 = tf.nn.relu(z1)
            z2 = tf.matmul(a1, W2) + b2
            a2 = tf.nn.relu(z2)
            z3 = tf.matmul(a2, W3) + b3
            return z3

        y_logits = forward_propagation(X)

        # Loss and Optimizer inside loop
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=Y))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Accuracy calculation inside loop
        correct_pred = tf.equal(tf.argmax(y_logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # New Session for each training
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            # Epoch loop
            for epoch in range(epochs):
                # Batch-wise training
                for i in range(0, x_train.shape[0], batch_size):
                    batch_x = x_train[i:i+batch_size]
                    batch_y = y_train_one_hot[i:i+batch_size]
                    sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

                # Compute training loss and accuracy after each epoch
                train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X: x_train, Y: y_train_one_hot})
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Testing the model
            test_acc, y_pred_logits = sess.run([accuracy, y_logits], feed_dict={X: x_test, Y: y_test_one_hot})
            execution_time = time.time() - start_time

            # Compute confusion matrix
            y_pred_classes = np.argmax(y_pred_logits, axis=1)
            conf_matrix = confusion_matrix(y_test, y_pred_classes)

            # Store results
            results[(batch_size, epochs)] = {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "test_accuracy": test_acc,
                "execution_time": execution_time,
                "conf_matrix": conf_matrix
            }

            print(f"Test Accuracy: {test_acc:.4f}, Execution Time: {execution_time:.2f} sec")

        # Plot Loss and Accuracy curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, label=f"Batch={batch_size}, Epochs={epochs}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_accuracies, label=f"Batch={batch_size}, Epochs={epochs}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()

        plt.show()

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix (Batch={batch_size}, Epochs={epochs})")
        plt.show()

# Print Execution Time Comparison
print("\nExecution Time Comparison:")
for key, val in results.items():
    print(f"Batch Size={key[0]}, Epochs={key[1]} -> Execution Time: {val['execution_time']:.2f} sec")

     
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step

Training for Batch Size=100

Training for 100 Epochs
Epoch 1/100, Loss: 0.1618, Accuracy: 0.9491
Epoch 2/100, Loss: 0.1190, Accuracy: 0.9638
Epoch 3/100, Loss: 0.0832, Accuracy: 0.9751
Epoch 4/100, Loss: 0.1121, Accuracy: 0.9679
Epoch 5/100, Loss: 0.0716, Accuracy: 0.9787
Epoch 6/100, Loss: 0.0871, Accuracy: 0.9748
Epoch 7/100, Loss: 0.0833, Accuracy: 0.9785
Epoch 8/100, Loss: 0.0795, Accuracy: 0.9790
Epoch 9/100, Loss: 0.0557, Accuracy: 0.9840
Epoch 10/100, Loss: 0.0583, Accuracy: 0.9847
Epoch 11/100, Loss: 0.0784, Accuracy: 0.9781
Epoch 12/100, Loss: 0.0571, Accuracy: 0.9851
Epoch 13/100, Loss: 0.0529, Accuracy: 0.9857
Epoch 14/100, Loss: 0.0564, Accuracy: 0.9861
Epoch 15/100, Loss: 0.0636, Accuracy: 0.9839
Epoch 16/100, Loss: 0.0645, Accuracy: 0.9861
Epoch 17/100, Loss: 0.0527, Accuracy: 0.9869
Epoch 18/100, Loss: 0.0533, Accuracy: 0.9878
Epoch 19/100, Loss: 0.0446, Accuracy: 0.9889
Epoch 20/100, Loss: 0.0403, Accuracy: 0.9892
Epoch 21/100, Loss: 0.0511, Accuracy: 0.9889
Epoch 22/100, Loss: 0.0536, Accuracy: 0.9897
Epoch 23/100, Loss: 0.0463, Accuracy: 0.9888
Epoch 24/100, Loss: 0.0381, Accuracy: 0.9902
Epoch 25/100, Loss: 0.0459, Accuracy: 0.9894
Epoch 26/100, Loss: 0.0387, Accuracy: 0.9905
Epoch 27/100, Loss: 0.0832, Accuracy: 0.9856
Epoch 28/100, Loss: 0.0301, Accuracy: 0.9926
Epoch 29/100, Loss: 0.0388, Accuracy: 0.9907
Epoch 30/100, Loss: 0.0410, Accuracy: 0.9903
Epoch 31/100, Loss: 0.0347, Accuracy: 0.9921
Epoch 32/100, Loss: 0.0820, Accuracy: 0.9843
Epoch 33/100, Loss: 0.0597, Accuracy: 0.9892
Epoch 34/100, Loss: 0.0364, Accuracy: 0.9924
Epoch 35/100, Loss: 0.0474, Accuracy: 0.9913
Epoch 36/100, Loss: 0.0240, Accuracy: 0.9946
Epoch 37/100, Loss: 0.0380, Accuracy: 0.9921
Epoch 38/100, Loss: 0.0364, Accuracy: 0.9922
Epoch 39/100, Loss: 0.1479, Accuracy: 0.9862
Epoch 40/100, Loss: 0.0406, Accuracy: 0.9916
Epoch 41/100, Loss: 0.0492, Accuracy: 0.9899
Epoch 42/100, Loss: 0.0339, Accuracy: 0.9925
Epoch 43/100, Loss: 0.0423, Accuracy: 0.9918
Epoch 44/100, Loss: 0.0228, Accuracy: 0.9947
Epoch 45/100, Loss: 0.0323, Accuracy: 0.9922
Epoch 46/100, Loss: 0.0641, Accuracy: 0.9904
Epoch 47/100, Loss: 0.0302, Accuracy: 0.9938
Epoch 48/100, Loss: 0.0234, Accuracy: 0.9952
Epoch 49/100, Loss: 0.0298, Accuracy: 0.9937
Epoch 50/100, Loss: 0.0505, Accuracy: 0.9926
Epoch 51/100, Loss: 0.0590, Accuracy: 0.9912
Epoch 52/100, Loss: 0.0272, Accuracy: 0.9931
Epoch 53/100, Loss: 0.0675, Accuracy: 0.9898
Epoch 54/100, Loss: 0.0359, Accuracy: 0.9935
Epoch 55/100, Loss: 0.0413, Accuracy: 0.9923
Epoch 56/100, Loss: 0.0281, Accuracy: 0.9945
Epoch 57/100, Loss: 0.0351, Accuracy: 0.9942
Epoch 58/100, Loss: 0.0292, Accuracy: 0.9940
Epoch 59/100, Loss: 0.0472, Accuracy: 0.9912
Epoch 60/100, Loss: 0.0255, Accuracy: 0.9945
Epoch 61/100, Loss: 0.0296, Accuracy: 0.9949
Epoch 62/100, Loss: 0.0307, Accuracy: 0.9938
Epoch 63/100, Loss: 0.0351, Accuracy: 0.9925
Epoch 64/100, Loss: 0.0359, Accuracy: 0.9933
Epoch 65/100, Loss: 0.0258, Accuracy: 0.9946
Epoch 66/100, Loss: 0.0446, Accuracy: 0.9913
Epoch 67/100, Loss: 0.0289, Accuracy: 0.9940
Epoch 68/100, Loss: 0.0323, Accuracy: 0.9929
Epoch 69/100, Loss: 0.0490, Accuracy: 0.9924
Epoch 70/100, Loss: 0.0345, Accuracy: 0.9937
Epoch 71/100, Loss: 0.0303, Accuracy: 0.9947
Epoch 72/100, Loss: 0.0177, Accuracy: 0.9962
Epoch 73/100, Loss: 0.0545, Accuracy: 0.9929
Epoch 74/100, Loss: 0.0460, Accuracy: 0.9934
Epoch 75/100, Loss: 0.0140, Accuracy: 0.9964
Epoch 76/100, Loss: 0.0219, Accuracy: 0.9947
Epoch 77/100, Loss: 0.0269, Accuracy: 0.9937
Epoch 78/100, Loss: 0.0325, Accuracy: 0.9933
Epoch 79/100, Loss: 0.0176, Accuracy: 0.9959
Epoch 80/100, Loss: 0.0821, Accuracy: 0.9865
Epoch 81/100, Loss: 0.0228, Accuracy: 0.9955
Epoch 82/100, Loss: 0.0319, Accuracy: 0.9940
Epoch 83/100, Loss: 0.0343, Accuracy: 0.9936
Epoch 84/100, Loss: 0.0217, Accuracy: 0.9949
Epoch 85/100, Loss: 0.0146, Accuracy: 0.9958
Epoch 86/100, Loss: 0.0399, Accuracy: 0.9898
Epoch 87/100, Loss: 0.0270, Accuracy: 0.9947
Epoch 88/100, Loss: 0.0330, Accuracy: 0.9908
Epoch 89/100, Loss: 0.0287, Accuracy: 0.9941
Epoch 90/100, Loss: 0.0725, Accuracy: 0.9898
Epoch 91/100, Loss: 0.0217, Accuracy: 0.9948
Epoch 92/100, Loss: 0.0205, Accuracy: 0.9956
Epoch 93/100, Loss: 0.0698, Accuracy: 0.9902
Epoch 94/100, Loss: 0.0637, Accuracy: 0.9909
Epoch 95/100, Loss: 0.0270, Accuracy: 0.9946
Epoch 96/100, Loss: 0.0206, Accuracy: 0.9945
Epoch 97/100, Loss: 0.0445, Accuracy: 0.9920
Epoch 98/100, Loss: 0.0594, Accuracy: 0.9901
Epoch 99/100, Loss: 0.0231, Accuracy: 0.9953
Epoch 100/100, Loss: 0.0212, Accuracy: 0.9951
Test Accuracy: 0.9730, Execution Time: 122.47 sec


Training for 50 Epochs
Epoch 1/50, Loss: 0.1535, Accuracy: 0.9539
Epoch 2/50, Loss: 0.1139, Accuracy: 0.9661
Epoch 3/50, Loss: 0.1635, Accuracy: 0.9554
Epoch 4/50, Loss: 0.1358, Accuracy: 0.9613
Epoch 5/50, Loss: 0.1346, Accuracy: 0.9664
Epoch 6/50, Loss: 0.1025, Accuracy: 0.9724
Epoch 7/50, Loss: 0.0769, Accuracy: 0.9790
Epoch 8/50, Loss: 0.0834, Accuracy: 0.9786
Epoch 9/50, Loss: 0.0570, Accuracy: 0.9833
Epoch 10/50, Loss: 0.0666, Accuracy: 0.9817
Epoch 11/50, Loss: 0.0672, Accuracy: 0.9835
Epoch 12/50, Loss: 0.0661, Accuracy: 0.9828
Epoch 13/50, Loss: 0.0788, Accuracy: 0.9797
Epoch 14/50, Loss: 0.0416, Accuracy: 0.9878
Epoch 15/50, Loss: 0.0546, Accuracy: 0.9856
Epoch 16/50, Loss: 0.0818, Accuracy: 0.9806
Epoch 17/50, Loss: 0.0577, Accuracy: 0.9866
Epoch 18/50, Loss: 0.0914, Accuracy: 0.9803
Epoch 19/50, Loss: 0.0650, Accuracy: 0.9871
Epoch 20/50, Loss: 0.0645, Accuracy: 0.9847
Epoch 21/50, Loss: 0.0561, Accuracy: 0.9866
Epoch 22/50, Loss: 0.0741, Accuracy: 0.9859
Epoch 23/50, Loss: 0.0790, Accuracy: 0.9831
Epoch 24/50, Loss: 0.0755, Accuracy: 0.9829
Epoch 25/50, Loss: 0.0731, Accuracy: 0.9843
Epoch 26/50, Loss: 0.0525, Accuracy: 0.9883
Epoch 27/50, Loss: 0.0386, Accuracy: 0.9905
Epoch 28/50, Loss: 0.0648, Accuracy: 0.9869
Epoch 29/50, Loss: 0.0605, Accuracy: 0.9887
Epoch 30/50, Loss: 0.0626, Accuracy: 0.9879
Epoch 31/50, Loss: 0.0453, Accuracy: 0.9905
Epoch 32/50, Loss: 0.0294, Accuracy: 0.9924
Epoch 33/50, Loss: 0.0438, Accuracy: 0.9904
Epoch 34/50, Loss: 0.0554, Accuracy: 0.9897
Epoch 35/50, Loss: 0.0518, Accuracy: 0.9897
Epoch 36/50, Loss: 0.0597, Accuracy: 0.9897
Epoch 37/50, Loss: 0.0492, Accuracy: 0.9908
Epoch 38/50, Loss: 0.0292, Accuracy: 0.9927
Epoch 39/50, Loss: 0.0536, Accuracy: 0.9893
Epoch 40/50, Loss: 0.0353, Accuracy: 0.9920
Epoch 41/50, Loss: 0.0401, Accuracy: 0.9915
Epoch 42/50, Loss: 0.0618, Accuracy: 0.9889
Epoch 43/50, Loss: 0.0333, Accuracy: 0.9920
Epoch 44/50, Loss: 0.0414, Accuracy: 0.9909
Epoch 45/50, Loss: 0.0389, Accuracy: 0.9920
Epoch 46/50, Loss: 0.0340, Accuracy: 0.9926
Epoch 47/50, Loss: 0.0487, Accuracy: 0.9918
Epoch 48/50, Loss: 0.0422, Accuracy: 0.9930
Epoch 49/50, Loss: 0.0527, Accuracy: 0.9914
Epoch 50/50, Loss: 0.0317, Accuracy: 0.9938
Test Accuracy: 0.9729, Execution Time: 57.29 sec


Training for 10 Epochs
Epoch 1/10, Loss: 0.1797, Accuracy: 0.9477
Epoch 2/10, Loss: 0.1323, Accuracy: 0.9620
Epoch 3/10, Loss: 0.1245, Accuracy: 0.9651
Epoch 4/10, Loss: 0.0750, Accuracy: 0.9775
Epoch 5/10, Loss: 0.0956, Accuracy: 0.9737
Epoch 6/10, Loss: 0.0808, Accuracy: 0.9772
Epoch 7/10, Loss: 0.0741, Accuracy: 0.9783
Epoch 8/10, Loss: 0.0877, Accuracy: 0.9772
Epoch 9/10, Loss: 0.0562, Accuracy: 0.9840
Epoch 10/10, Loss: 0.0887, Accuracy: 0.9796
Test Accuracy: 0.9674, Execution Time: 11.78 sec


Training for Batch Size=10

Training for 100 Epochs
Epoch 1/100, Loss: 0.3012, Accuracy: 0.9299
Epoch 2/100, Loss: 0.8656, Accuracy: 0.8605
Epoch 3/100, Loss: 0.1917, Accuracy: 0.9536
Epoch 4/100, Loss: 0.2427, Accuracy: 0.9472
Epoch 5/100, Loss: 0.2826, Accuracy: 0.9481
Epoch 6/100, Loss: 0.2237, Accuracy: 0.9576
Epoch 7/100, Loss: 0.2250, Accuracy: 0.9406
Epoch 8/100, Loss: 0.2900, Accuracy: 0.9320
Epoch 9/100, Loss: 0.1791, Accuracy: 0.9607
Epoch 10/100, Loss: 0.2496, Accuracy: 0.9427
Epoch 11/100, Loss: 0.2330, Accuracy: 0.9574
Epoch 12/100, Loss: 0.2711, Accuracy: 0.9451
Epoch 13/100, Loss: 0.2505, Accuracy: 0.9275
Epoch 14/100, Loss: 0.2321, Accuracy: 0.9528
Epoch 15/100, Loss: 0.1956, Accuracy: 0.9560
Epoch 16/100, Loss: 0.2451, Accuracy: 0.9487
Epoch 17/100, Loss: 0.2959, Accuracy: 0.9371
Epoch 18/100, Loss: 0.2346, Accuracy: 0.9493
Epoch 19/100, Loss: 0.2260, Accuracy: 0.9465
Epoch 20/100, Loss: 0.3228, Accuracy: 0.9280
Epoch 21/100, Loss: 0.2801, Accuracy: 0.9384
Epoch 22/100, Loss: 0.4165, Accuracy: 0.9257
Epoch 23/100, Loss: 0.2906, Accuracy: 0.9327
Epoch 24/100, Loss: 0.2713, Accuracy: 0.9116
Epoch 25/100, Loss: 0.3572, Accuracy: 0.8993
Epoch 26/100, Loss: 0.3630, Accuracy: 0.9242
Epoch 27/100, Loss: 0.3298, Accuracy: 0.9040
Epoch 28/100, Loss: 0.3267, Accuracy: 0.8975
Epoch 29/100, Loss: 0.2827, Accuracy: 0.9233
Epoch 30/100, Loss: 0.4343, Accuracy: 0.8985
Epoch 31/100, Loss: 0.4457, Accuracy: 0.9146
Epoch 32/100, Loss: 0.4639, Accuracy: 0.9168
Epoch 33/100, Loss: 0.3564, Accuracy: 0.9013
Epoch 34/100, Loss: 0.4808, Accuracy: 0.9191
Epoch 35/100, Loss: 0.2546, Accuracy: 0.9184
Epoch 36/100, Loss: 0.3877, Accuracy: 0.8868
Epoch 37/100, Loss: 0.3373, Accuracy: 0.8967
Epoch 38/100, Loss: 0.4611, Accuracy: 0.9026
Epoch 39/100, Loss: 1.0364, Accuracy: 0.8738
Epoch 40/100, Loss: 0.4023, Accuracy: 0.9006
Epoch 41/100, Loss: 0.4179, Accuracy: 0.9136
Epoch 42/100, Loss: 0.4958, Accuracy: 0.8578
Epoch 43/100, Loss: 0.6771, Accuracy: 0.8748
Epoch 44/100, Loss: 0.4735, Accuracy: 0.8478
Epoch 45/100, Loss: 0.3977, Accuracy: 0.8608
Epoch 46/100, Loss: 0.4894, Accuracy: 0.8375
Epoch 47/100, Loss: 0.4036, Accuracy: 0.8564
Epoch 48/100, Loss: 0.4904, Accuracy: 0.8397
Epoch 49/100, Loss: 0.5311, Accuracy: 0.8733
Epoch 50/100, Loss: 1.4862, Accuracy: 0.8799
Epoch 51/100, Loss: 0.7049, Accuracy: 0.8615
Epoch 52/100, Loss: 0.4614, Accuracy: 0.8557
Epoch 53/100, Loss: 0.5596, Accuracy: 0.8352
Epoch 54/100, Loss: 0.5364, Accuracy: 0.8429
Epoch 55/100, Loss: 0.5032, Accuracy: 0.8780
Epoch 56/100, Loss: 0.5996, Accuracy: 0.8529
Epoch 57/100, Loss: 0.5848, Accuracy: 0.8632
Epoch 58/100, Loss: 0.9157, Accuracy: 0.8554
Epoch 59/100, Loss: 1.8390, Accuracy: 0.8541
Epoch 60/100, Loss: 0.5837, Accuracy: 0.8194
Epoch 61/100, Loss: 0.5505, Accuracy: 0.8337
Epoch 62/100, Loss: 0.8974, Accuracy: 0.8618
Epoch 63/100, Loss: 0.5140, Accuracy: 0.8429
Epoch 64/100, Loss: 0.6086, Accuracy: 0.8490
Epoch 65/100, Loss: 0.6682, Accuracy: 0.7685
Epoch 66/100, Loss: 0.6400, Accuracy: 0.8245
Epoch 67/100, Loss: 0.6223, Accuracy: 0.7745
Epoch 68/100, Loss: 0.7459, Accuracy: 0.8224
Epoch 69/100, Loss: 0.7495, Accuracy: 0.7882
Epoch 70/100, Loss: 0.5891, Accuracy: 0.7983
Epoch 71/100, Loss: 0.9803, Accuracy: 0.8699
Epoch 72/100, Loss: 2.7941, Accuracy: 0.7801
Epoch 73/100, Loss: 0.5658, Accuracy: 0.8168
Epoch 74/100, Loss: 0.7416, Accuracy: 0.7843
Epoch 75/100, Loss: 0.7595, Accuracy: 0.7661
Epoch 76/100, Loss: 0.8028, Accuracy: 0.8008
Epoch 77/100, Loss: 0.6946, Accuracy: 0.7631
Epoch 78/100, Loss: 0.7470, Accuracy: 0.7436
Epoch 79/100, Loss: 0.6446, Accuracy: 0.7717
Epoch 80/100, Loss: 0.7333, Accuracy: 0.7842
Epoch 81/100, Loss: 0.7861, Accuracy: 0.7469
Epoch 82/100, Loss: 1.0327, Accuracy: 0.7597
Epoch 83/100, Loss: 0.7904, Accuracy: 0.7191
Epoch 84/100, Loss: 0.8444, Accuracy: 0.7159
Epoch 85/100, Loss: 0.8983, Accuracy: 0.7165
Epoch 86/100, Loss: 1.2816, Accuracy: 0.7827
Epoch 87/100, Loss: 0.9375, Accuracy: 0.7527
Epoch 88/100, Loss: 0.6744, Accuracy: 0.7708
Epoch 89/100, Loss: 1.2370, Accuracy: 0.7614
Epoch 90/100, Loss: 0.8642, Accuracy: 0.7103
Epoch 91/100, Loss: 0.8994, Accuracy: 0.7374
Epoch 92/100, Loss: 0.8753, Accuracy: 0.7871
Epoch 93/100, Loss: 0.9972, Accuracy: 0.7142
Epoch 94/100, Loss: 0.8664, Accuracy: 0.7492
Epoch 95/100, Loss: 1.0593, Accuracy: 0.7041
Epoch 96/100, Loss: 0.7142, Accuracy: 0.7508
Epoch 97/100, Loss: 0.8341, Accuracy: 0.6832
Epoch 98/100, Loss: 2.8257, Accuracy: 0.7281
Epoch 99/100, Loss: 0.9082, Accuracy: 0.6699
Epoch 100/100, Loss: 1.7557, Accuracy: 0.7657
Test Accuracy: 0.7641, Execution Time: 767.64 sec


Training for 50 Epochs
Epoch 1/50, Loss: 0.3548, Accuracy: 0.9130
Epoch 2/50, Loss: 0.2344, Accuracy: 0.9403
Epoch 3/50, Loss: 0.2369, Accuracy: 0.9422
Epoch 4/50, Loss: 0.2418, Accuracy: 0.9416
Epoch 5/50, Loss: 0.2714, Accuracy: 0.9454
Epoch 6/50, Loss: 0.2337, Accuracy: 0.9552
Epoch 7/50, Loss: 0.2054, Accuracy: 0.9572
Epoch 8/50, Loss: 0.2764, Accuracy: 0.9431
Epoch 9/50, Loss: 0.2007, Accuracy: 0.9530
Epoch 10/50, Loss: 0.2002, Accuracy: 0.9460
Epoch 11/50, Loss: 0.2034, Accuracy: 0.9599
Epoch 12/50, Loss: 0.2298, Accuracy: 0.9444
Epoch 13/50, Loss: 0.1807, Accuracy: 0.9558
Epoch 14/50, Loss: 0.2900, Accuracy: 0.9471
Epoch 15/50, Loss: 0.1857, Accuracy: 0.9510
Epoch 16/50, Loss: 0.4334, Accuracy: 0.9270
Epoch 17/50, Loss: 0.2340, Accuracy: 0.9497
Epoch 18/50, Loss: 0.3220, Accuracy: 0.9159
Epoch 19/50, Loss: 0.3128, Accuracy: 0.9418
Epoch 20/50, Loss: 0.2361, Accuracy: 0.9332
Epoch 21/50, Loss: 0.3287, Accuracy: 0.9060
Epoch 22/50, Loss: 0.3153, Accuracy: 0.9278
Epoch 23/50, Loss: 0.2617, Accuracy: 0.9139
Epoch 24/50, Loss: 0.4231, Accuracy: 0.9222
Epoch 25/50, Loss: 0.3265, Accuracy: 0.9392
Epoch 26/50, Loss: 0.4219, Accuracy: 0.9170
Epoch 27/50, Loss: 0.4053, Accuracy: 0.9254
Epoch 28/50, Loss: 0.4284, Accuracy: 0.9241
Epoch 29/50, Loss: 0.3269, Accuracy: 0.8874
Epoch 30/50, Loss: 0.3511, Accuracy: 0.9353
Epoch 31/50, Loss: 0.4218, Accuracy: 0.8849
Epoch 32/50, Loss: 0.3446, Accuracy: 0.8972
Epoch 33/50, Loss: 0.4526, Accuracy: 0.8690
Epoch 34/50, Loss: 0.3337, Accuracy: 0.8992
Epoch 35/50, Loss: 0.3445, Accuracy: 0.8996
Epoch 36/50, Loss: 0.3603, Accuracy: 0.9019
Epoch 37/50, Loss: 0.4013, Accuracy: 0.8979
Epoch 38/50, Loss: 0.3987, Accuracy: 0.8965
Epoch 39/50, Loss: 0.4367, Accuracy: 0.8792
Epoch 40/50, Loss: 0.3612, Accuracy: 0.8900
Epoch 41/50, Loss: 0.4123, Accuracy: 0.8778
Epoch 42/50, Loss: 0.4141, Accuracy: 0.8655
Epoch 43/50, Loss: 0.3868, Accuracy: 0.8698
Epoch 44/50, Loss: 0.4691, Accuracy: 0.8646
Epoch 45/50, Loss: 0.4592, Accuracy: 0.8451
Epoch 46/50, Loss: 0.3850, Accuracy: 0.8748
Epoch 47/50, Loss: 0.5839, Accuracy: 0.8186
Epoch 48/50, Loss: 0.4164, Accuracy: 0.8651
Epoch 49/50, Loss: 0.4525, Accuracy: 0.8945
Epoch 50/50, Loss: 0.4385, Accuracy: 0.8655
Test Accuracy: 0.8623, Execution Time: 383.03 sec


Training for 10 Epochs
Epoch 1/10, Loss: 0.2446, Accuracy: 0.9391
Epoch 2/10, Loss: 0.2478, Accuracy: 0.9429
Epoch 3/10, Loss: 0.2282, Accuracy: 0.9475
Epoch 4/10, Loss: 0.1878, Accuracy: 0.9590
Epoch 5/10, Loss: 0.2603, Accuracy: 0.9488
Epoch 6/10, Loss: 0.2092, Accuracy: 0.9589
Epoch 7/10, Loss: 0.2821, Accuracy: 0.9417
Epoch 8/10, Loss: 0.2502, Accuracy: 0.9372
Epoch 9/10, Loss: 0.1761, Accuracy: 0.9584
Epoch 10/10, Loss: 0.2783, Accuracy: 0.8569
Test Accuracy: 0.8548, Execution Time: 76.73 sec


Training for Batch Size=1

Training for 100 Epochs
Epoch 1/100, Loss: 1.3785, Accuracy: 0.5726
Epoch 2/100, Loss: 1.8956, Accuracy: 0.2749
Epoch 3/100, Loss: 1.9756, Accuracy: 0.2481
Epoch 4/100, Loss: 2.2134, Accuracy: 0.1386
Epoch 5/100, Loss: 6.7257, Accuracy: 0.2487
Epoch 6/100, Loss: 2.0420, Accuracy: 0.2086
Epoch 7/100, Loss: 2.1828, Accuracy: 0.1495
Epoch 8/100, Loss: 2.0499, Accuracy: 0.1960
Epoch 9/100, Loss: 2.0471, Accuracy: 0.1918
Epoch 10/100, Loss: 2.0836, Accuracy: 0.2042
Epoch 11/100, Loss: 2.1188, Accuracy: 0.2049
Epoch 12/100, Loss: 2.2134, Accuracy: 0.2172
Epoch 13/100, Loss: 2.0135, Accuracy: 0.2067
Epoch 14/100, Loss: 2.2727, Accuracy: 0.1976
Epoch 15/100, Loss: 2.0151, Accuracy: 0.2110
Epoch 16/100, Loss: 2.2827, Accuracy: 0.2060
Epoch 17/100, Loss: 2.0486, Accuracy: 0.1948
Epoch 18/100, Loss: 2.0621, Accuracy: 0.2001
Epoch 19/100, Loss: 2.6116, Accuracy: 0.2088
Epoch 20/100, Loss: 2.0416, Accuracy: 0.1976
Epoch 21/100, Loss: 2.1189, Accuracy: 0.2188
Epoch 22/100, Loss: 2.0723, Accuracy: 0.1905
Epoch 23/100, Loss: 2.0532, Accuracy: 0.1983
Epoch 24/100, Loss: 2.0785, Accuracy: 0.2058
Epoch 25/100, Loss: 2.0636, Accuracy: 0.2198
Epoch 26/100, Loss: 2.0780, Accuracy: 0.1856
Epoch 27/100, Loss: 2.0340, Accuracy: 0.2002
Epoch 28/100, Loss: 2.0584, Accuracy: 0.1924
Epoch 29/100, Loss: 2.1583, Accuracy: 0.1583
Epoch 30/100, Loss: 2.7005, Accuracy: 0.2244
Epoch 31/100, Loss: 1.9898, Accuracy: 0.2133
Epoch 32/100, Loss: 1.9871, Accuracy: 0.2150
Epoch 33/100, Loss: 2.0757, Accuracy: 0.2099
Epoch 34/100, Loss: 2.0079, Accuracy: 0.2115
Epoch 35/100, Loss: 2.2917, Accuracy: 0.2246
Epoch 36/100, Loss: 2.0267, Accuracy: 0.2011
Epoch 37/100, Loss: 2.0084, Accuracy: 0.2057
Epoch 38/100, Loss: 2.0158, Accuracy: 0.2045
Epoch 39/100, Loss: 2.1185, Accuracy: 0.1751
Epoch 40/100, Loss: 2.0325, Accuracy: 0.1990
Epoch 41/100, Loss: 2.0506, Accuracy: 0.1959
Epoch 42/100, Loss: 2.0200, Accuracy: 0.2044
Epoch 43/100, Loss: 1.9804, Accuracy: 0.2158
Epoch 44/100, Loss: 2.1468, Accuracy: 0.1625
Epoch 45/100, Loss: 1.9869, Accuracy: 0.2111
Epoch 46/100, Loss: 1.9823, Accuracy: 0.2136
Epoch 47/100, Loss: 1.9730, Accuracy: 0.2195
Epoch 48/100, Loss: 2.1879, Accuracy: 0.2213
Epoch 49/100, Loss: 2.0446, Accuracy: 0.1952
Epoch 50/100, Loss: 2.2352, Accuracy: 0.2118
Epoch 51/100, Loss: 2.0151, Accuracy: 0.2035
Epoch 52/100, Loss: 1.9943, Accuracy: 0.2087
Epoch 53/100, Loss: 2.0954, Accuracy: 0.1804
Epoch 54/100, Loss: 2.0188, Accuracy: 0.2031
Epoch 55/100, Loss: 2.0353, Accuracy: 0.1993
Epoch 56/100, Loss: 1.9713, Accuracy: 0.2178
Epoch 57/100, Loss: 2.5022, Accuracy: 0.2162
Epoch 58/100, Loss: 2.0940, Accuracy: 0.1816
Epoch 59/100, Loss: 2.0362, Accuracy: 0.1991
Epoch 60/100, Loss: 2.0045, Accuracy: 0.2074
Epoch 61/100, Loss: 1.9740, Accuracy: 0.2192
Epoch 62/100, Loss: 2.0013, Accuracy: 0.2080
Epoch 63/100, Loss: 2.0078, Accuracy: 0.2065
Epoch 64/100, Loss: 7.1340, Accuracy: 0.2212
Epoch 65/100, Loss: 1.9744, Accuracy: 0.2170
Epoch 66/100, Loss: 1.9910, Accuracy: 0.2099
Epoch 67/100, Loss: 1.9945, Accuracy: 0.2119
Epoch 68/100, Loss: 2.0910, Accuracy: 0.2075
Epoch 69/100, Loss: 2.2749, Accuracy: 0.2348
Epoch 70/100, Loss: 2.0043, Accuracy: 0.2129
Epoch 71/100, Loss: 2.1210, Accuracy: 0.1750
Epoch 72/100, Loss: 2.0802, Accuracy: 0.1906
Epoch 73/100, Loss: 2.2649, Accuracy: 0.2265
Epoch 74/100, Loss: 2.2215, Accuracy: 0.2037
Epoch 75/100, Loss: 2.0913, Accuracy: 0.1822
Epoch 76/100, Loss: 2.0898, Accuracy: 0.1825
Epoch 77/100, Loss: 2.0316, Accuracy: 0.2004
Epoch 78/100, Loss: 1.9680, Accuracy: 0.2200
Epoch 79/100, Loss: 1.9828, Accuracy: 0.2156
Epoch 80/100, Loss: 2.0241, Accuracy: 0.2046
Epoch 81/100, Loss: 2.0176, Accuracy: 0.2076
Epoch 82/100, Loss: 2.3942, Accuracy: 0.2285
Epoch 83/100, Loss: 1.9729, Accuracy: 0.2242
Epoch 84/100, Loss: 2.1568, Accuracy: 0.2176
Epoch 85/100, Loss: 2.0351, Accuracy: 0.2001
Epoch 86/100, Loss: 2.0044, Accuracy: 0.2147
Epoch 87/100, Loss: 1.9651, Accuracy: 0.2212
Epoch 88/100, Loss: 1.9644, Accuracy: 0.2259
Epoch 89/100, Loss: 2.0006, Accuracy: 0.2116
Epoch 90/100, Loss: 1.9867, Accuracy: 0.2164
Epoch 91/100, Loss: 1.9548, Accuracy: 0.2229
Epoch 92/100, Loss: 2.0062, Accuracy: 0.2194
Epoch 93/100, Loss: 2.7764, Accuracy: 0.2184
Epoch 94/100, Loss: 2.4207, Accuracy: 0.2225
Epoch 95/100, Loss: 2.0080, Accuracy: 0.2087
Epoch 96/100, Loss: 1.9622, Accuracy: 0.2195
Epoch 97/100, Loss: 1.9546, Accuracy: 0.2227
Epoch 98/100, Loss: 2.3287, Accuracy: 0.2353
Epoch 99/100, Loss: 1.9560, Accuracy: 0.2216
Epoch 100/100, Loss: 2.1173, Accuracy: 0.2266
Test Accuracy: 0.2282, Execution Time: 7058.97 sec


Training for 50 Epochs
Epoch 1/50, Loss: 1.4926, Accuracy: 0.5145
Epoch 2/50, Loss: 2.1873, Accuracy: 0.3089
Epoch 3/50, Loss: 2.1123, Accuracy: 0.1700
Epoch 4/50, Loss: 2.0502, Accuracy: 0.2064
Epoch 5/50, Loss: 4.6746, Accuracy: 0.2823
Epoch 6/50, Loss: 1.9542, Accuracy: 0.2480
Epoch 7/50, Loss: 2.1582, Accuracy: 0.1675
Epoch 8/50, Loss: 2.2093, Accuracy: 0.2610
Epoch 9/50, Loss: 2.1554, Accuracy: 0.1649
Epoch 10/50, Loss: 4.0962, Accuracy: 0.3006
Epoch 11/50, Loss: 2.0296, Accuracy: 0.2252
Epoch 12/50, Loss: 2.2400, Accuracy: 0.2184
Epoch 13/50, Loss: 3.6558, Accuracy: 0.2620
Epoch 14/50, Loss: 2.0955, Accuracy: 0.1855
Epoch 15/50, Loss: 2.1883, Accuracy: 0.2615
Epoch 16/50, Loss: 1.9820, Accuracy: 0.2308
Epoch 17/50, Loss: 1.9622, Accuracy: 0.2402
Epoch 18/50, Loss: 2.0269, Accuracy: 0.2137
Epoch 19/50, Loss: 2.0122, Accuracy: 0.2240
Epoch 20/50, Loss: 2.8323, Accuracy: 0.1979
Epoch 21/50, Loss: 3.1004, Accuracy: 0.2861
Epoch 22/50, Loss: 2.3613, Accuracy: 0.2437
Epoch 23/50, Loss: 2.2248, Accuracy: 0.2961
Epoch 24/50, Loss: 2.3593, Accuracy: 0.2233
Epoch 25/50, Loss: 2.0500, Accuracy: 0.2771
Epoch 26/50, Loss: 2.0885, Accuracy: 0.2814
Epoch 27/50, Loss: 2.0138, Accuracy: 0.2435
Epoch 28/50, Loss: 1.9653, Accuracy: 0.2605
Epoch 29/50, Loss: 1.9544, Accuracy: 0.2292
Epoch 30/50, Loss: 1.9319, Accuracy: 0.2590
Epoch 31/50, Loss: 2.0965, Accuracy: 0.2595
Epoch 32/50, Loss: 2.0956, Accuracy: 0.1866
Epoch 33/50, Loss: 2.0097, Accuracy: 0.2187
Epoch 34/50, Loss: 1.9349, Accuracy: 0.2606
Epoch 35/50, Loss: 2.0084, Accuracy: 0.2150
Epoch 36/50, Loss: 2.1352, Accuracy: 0.1586
Epoch 37/50, Loss: 2.2304, Accuracy: 0.2301
Epoch 38/50, Loss: 2.0479, Accuracy: 0.2030
Epoch 39/50, Loss: 1.9509, Accuracy: 0.2557
Epoch 40/50, Loss: 2.1596, Accuracy: 0.2619
Epoch 41/50, Loss: 1.9047, Accuracy: 0.2615
Epoch 42/50, Loss: 1.9786, Accuracy: 0.2247
Epoch 43/50, Loss: 2.1335, Accuracy: 0.2653
Epoch 44/50, Loss: 2.1799, Accuracy: 0.2781
Epoch 45/50, Loss: 2.0756, Accuracy: 0.2839
Epoch 46/50, Loss: 2.6120, Accuracy: 0.2908
Epoch 47/50, Loss: 1.9941, Accuracy: 0.2255
Epoch 48/50, Loss: 1.9025, Accuracy: 0.2621
Epoch 49/50, Loss: 10.6897, Accuracy: 0.3028
Epoch 50/50, Loss: 1.8695, Accuracy: 0.2649
Test Accuracy: 0.2670, Execution Time: 3524.98 sec


Training for 10 Epochs
Epoch 1/10, Loss: 0.8711, Accuracy: 0.7811
Epoch 2/10, Loss: 1.3488, Accuracy: 0.6791
Epoch 3/10, Loss: 1.9818, Accuracy: 0.4814
Epoch 4/10, Loss: 1.8546, Accuracy: 0.2931
Epoch 5/10, Loss: 2.0351, Accuracy: 0.2137
Epoch 6/10, Loss: 2.0556, Accuracy: 0.2421
Epoch 7/10, Loss: 1.9957, Accuracy: 0.2687
Epoch 8/10, Loss: 3.1793, Accuracy: 0.3039
Epoch 9/10, Loss: 1.9070, Accuracy: 0.2592
Epoch 10/10, Loss: 1.9501, Accuracy: 0.2746
Test Accuracy: 0.2737, Execution Time: 697.63 sec


Execution Time Comparison:
Batch Size=100, Epochs=100 -> Execution Time: 122.47 sec
Batch Size=100, Epochs=50 -> Execution Time: 57.29 sec
Batch Size=100, Epochs=10 -> Execution Time: 11.78 sec
Batch Size=10, Epochs=100 -> Execution Time: 767.64 sec
Batch Size=10, Epochs=50 -> Execution Time: 383.03 sec
Batch Size=10, Epochs=10 -> Execution Time: 76.73 sec
Batch Size=1, Epochs=100 -> Execution Time: 7058.97 sec
Batch Size=1, Epochs=50 -> Execution Time: 3524.98 sec
Batch Size=1, Epochs=10 -> Execution Time: 697.63 sec
Code Explanation:
1️⃣ Importing Libraries:

TensorFlow: For building and training the neural network.
NumPy: For handling arrays and numerical operations.
Matplotlib & Seaborn: For plotting and visualization.
Scikit-learn: For computing the confusion matrix.
2️⃣ Load the MNIST Dataset:

Eager execution is disabled to make the code compatible with TensorFlow 1.x.
MNIST dataset is loaded, which consists of 28×28 grayscale images of handwritten digits (0-9).
3️⃣ Data Preprocessing:

Normalization: The pixel values are scaled to [0,1] for better training stability.
Reshaping: The 28×28 images are flattened into 1D vectors of size 784 pixels.
One-hot encoding: Converts class labels (0-9) into a categorical format (e.g., 5 becomes [0,0,0,0,0,1,0,0,0,0]).
4️⃣ Training Configurations:

The training will be performed for three different batch sizes (100, 10, 1) and three different epochs (100, 50, 10).
5️⃣ Training Loop for Different Batch Sizes & Epochs:

The model is trained multiple times for different combinations of batch sizes and epochs.
6️⃣ Neural Network Initialization:

Resets the TensorFlow graph to avoid carrying over variables from the previous training runs.
Defines placeholders for input data (X) and labels (Y).
A helper function to initialize weights with a small random value.
Three-layer neural network:
Input layer (784) → Hidden layer (128) → Hidden layer (64) → Output layer (10).
7️⃣ Forward Propagation:

Activation function: ReLU (tf.nn.relu) is used in hidden layers.
Final output (logits) is computed.
8️⃣ Loss Function & Optimizer:

Loss function: Softmax Cross Entropy (since it's a multi-class classification problem).
Optimizer: Adam Optimizer (tf.compat.v1.train.AdamOptimizer).
Accuracy calculation: Compares the predicted and actual labels.
9️⃣ Training the Model:

Starts a new TensorFlow session and initializes all variables.
Implements Mini-batch Gradient Descent.
Computes training loss and accuracy.
🔟 Testing & Confusion Matrix:

Computes test accuracy.
Generates a Confusion Matrix to evaluate the model's performance.
1️⃣1️⃣ Plotting Loss & Accuracy Curves:

Plots the Loss and Accuracy Curve.
1️⃣2️⃣ Confusion Matrix Visualization:

Displays the Confusion Matrix as a Heatmap.
1️⃣3️⃣ Execution Time Comparison:

Compares the execution time for different batch sizes and epochs.
My Comments:
✅ Reducing the learning rate (e.g., 0.01 or 0.001) can help improve accuracy.

✅ Increasing the number of hidden layers may improve model accuracy.

✅ Using batch size 1 is not needed because it only increases training time without much benefit.

✅ Choosing the right batch size and number of epochs can improve model efficiency.

✅ Analyzing loss curves and accuracy trends can help fine-tune the training process.

✅ Execution time is much lower for larger batch sizes.

✅ Training with Batch Size = 1 took more time but did not improve accuracy.
