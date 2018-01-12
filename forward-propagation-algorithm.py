# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Declare W1, W2 two variables. Here, the random seeds are set by the seed parameter,
# this ensures that the results of each run are the same.
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

'''
# For the time being, define the input eigenvector as a constant. Note that x is a 1*2 matrix
x = tf.constant([[0.7, 0.9]])
'''

# Define placeholder as a place to store input data. Dimensions do not have to be defined here.
# But if the dimension is determined, the given dimension can reduce the probability of error.
# x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

# The output of neural network is obtained by forward propagation algorithm.
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

'''
sess.run(w1.initializer)    # Initialization of W1
sess.run(w2.initializer)    # Initialization w2
'''

init_op = tf.global_variables_initializer()
sess.run(init_op)

# Output : 3.95757794
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

sess.close()
