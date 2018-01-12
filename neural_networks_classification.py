import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from numpy.random import RandomState

# Defines the size of the training data batch.
batch_size = 8

# Defining the parameters of the neural network.
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# Using None on one dimension of shape makes it easier to use small batch sizes.
# You need to divide the data into smaller batches during training, but when testing. You can use all the data at once.
# It is easy to test when the data set is small, but when the data set is large,
# putting a large amount of data into a batch may result in memory overflow.
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# Defining the forward Propagation process of Neural Networks.
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# An algorithm for defining loss functions and backpropagation.
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# Generating an analog dataset from random numbers
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# Define rules to label samples. In this case, all samples x1 x 2 < 1 are considered positive samples
# (for example, parts are qualified) and others are negative samples (such as parts that are not qualified).
# Here 0 is used to represent the negative sample 1 to represent the positive sample.
# Most neural networks that solve the classification problem use the representation of 0 and 1.
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# Create a session to run the TensorFlow program.
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # Initialize variables.
    sess.run(init_op)
    print sess.run(w1)
    print sess.run(w2)

    # Sets the number of rotations of the train.
    STEPS = 5000
    for i in range(STEPS):
        # Batch_size samples were selected for training each time.
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # The neural network is trained and parameters are updated by the selected sample .
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # Cross entropy and output on all data every other time.
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))

    print sess.run(w1)
    print sess.run(w2)