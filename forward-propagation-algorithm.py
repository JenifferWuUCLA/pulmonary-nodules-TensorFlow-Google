import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Declare W1, W2 two variables. Here, the random seeds are set by the seed parameter,
# this ensures that the results of each run are the same.
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# For the time being, define the input eigenvector as a constant. Note that x is a 1*2 matrix
x = tf.constant([[0.7, 0.9]])

# The output of neural network is obtained by forward propagation algorithm.
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
sess.run(w1.initializer)    # Initialization of W1
sess.run(w2.initializer)    # Initialization w2
# Output : 3.95757794
print(sess.run(y))
sess.close()
