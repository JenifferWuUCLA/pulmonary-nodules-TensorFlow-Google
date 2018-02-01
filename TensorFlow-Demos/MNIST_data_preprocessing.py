# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data

# Loads the MNIST dataset. If you specify the address /home/jenifferwu/TensorFlow_data/MNIST_data,
# then TensorFlow automatically downloads the data.
mnist = input_data.read_data_sets("/home/jenifferwu/TensorFlow_data/MNIST_data", one_hot=True)

# Print Training data size: 55000.
print "Training data size: ", mnist.train.num_examples

# Print Validating data size: 5000.
print "Validating data size: ", mnist.validation.num_examples

# Print Testing size: 10000. data
print "Testing data size: ", mnist.test.num_examples

# Print Example training data: [0. 0. 0. ... 0.380 0.376 ... 0.]
print "Example training data: ", mnist.train.images[0]

# Print Example training data label:
# [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print "Examples training data label: ", mnist.train.labels[0]