import numpy as np
import tensorflow as tf
import loading_binary_data as data
import matplotlib.pyplot as plt

training = data.train("emnist/emnist-binary/")
print(training)

# using two numpy arrays
features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))

iter = training.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))