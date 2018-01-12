import Tensorflow.mnist.input_data as input_data
import numpy as np

ROWs = 28
COLs = 28
NUM_LABELs = 10

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

