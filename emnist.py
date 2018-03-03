# Code adapted from: https://github.com/j05t/emnist/blob/master/emnist.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as spio
import keras

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

# allocate only as much GPU memory based on runtime allocations
config.gpu_options.allow_growth = True

emnist = spio.loadmat("emnist/emnist-matlab/emnist-balanced.mat")

# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

# store labels for visualization
train_labels = y_train
test_labels = y_test

#print(x_train.shape)
#print(y_train.shape)

# normalize
x_train /= 255
x_test /= 255

#print(x_train)
#print(x_train.shape)
# # reshape using matlab order
# x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
# x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

#print(x_train.shape)

#print(x_train.shape)  # (112800, 1, 28, 28)
#print(y_train.shape)  # (112800, 1)

# labels should be onehot encoded
y_train = tf.keras.utils.to_categorical(y_train, 47)
y_test = tf.keras.utils.to_categorical(y_test, 47)

#y_train = keras.utils.to_categorical(y_train, 47)
#y_test = keras.utils.to_categorical(y_test, 47)

#print(y_test.shape)  # (18800, 47)
#print(y_train.shape)  # (112800, 47)

# which sample to look at
samplenum = 5436

img = x_train[samplenum]

# visualize image
#im = plt.imshow(img[0])
#plt.show()

# show label for sample image
#print(train_labels[samplenum][0])

# Reshape labels (reduces dimensionality)
#print(test_labels.shape)  # (18800, 47)
#print(train_labels.shape)  # (112800, 1)

test_labels = test_labels.reshape(test_labels.shape[0])
train_labels = train_labels.reshape(train_labels.shape[0])

#print(test_labels.shape)  # (18800,)
#print(train_labels.shape)  # (112800,)

# Parameters
learning_rate = 0.01
momentum = 0.5
training_epochs = 300
batch_size = 500
display_step = 1


def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0 / weight_shape[0]) ** 0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [784, 256], [256])

    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [256, 256], [256])

    with tf.variable_scope("output"):
        output = layer(hidden_2, [256, 47], [47])

    return output


def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss


def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


with tf.Graph().as_default():

    # mnist data image of shape 28*28=784
    x = tf.placeholder("float", [None, 784])

    # 0-9 digits recognition => 10
    # 47 classes
    y = tf.placeholder("float", [None, 47])

    output = inference(x)
    cost = loss(output, y)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)
    eval_op = evaluate(output, y)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter("logs/", graph=sess.graph)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("Total Training Samples:", x_train.shape[0])

    # Training Cycle
    for epoch in range(training_epochs):
        print("Epoch: " + str(epoch) + "/" + str(training_epochs))

        avg_cost = 0.
        total_batch = int(x_train.shape[0]/batch_size)

        start_batch = 0
        end_batch = batch_size
        # Loop over all batches
        for i in range(total_batch):
            #mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            mbatch_x = x_train[start_batch:end_batch]
            mbatch_y = y_train[start_batch:end_batch]

            # Fit training using batch data
            feed_dict = {x: mbatch_x, y: mbatch_y}
            sess.run(train_op, feed_dict=feed_dict)

            # Compute average loss
            minibatch_cost = sess.run(cost, feed_dict=feed_dict)
            avg_cost += minibatch_cost/total_batch
            print("Batch:", str(i) + "/" + str(total_batch) + " - Average Loss: " + str(avg_cost), end='\r')

            # keep track of next batch
            start_batch = end_batch
            end_batch += batch_size
            if end_batch > total_batch:
                end_batch = total_batch

        print("Batch:", str(i) + "/" + str(total_batch) + " - Average Loss: " + str(avg_cost))

        # Display logs per epoch step
        if epoch % display_step == 0:
            validation_images = x_train[:5000]
            validation_labels = y_train[:5000]
            val_feed_dict = {
                x: x_test,
                y: y_test
            }
            accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
            print("Epoch: ", str(epoch) + "/" + str(training_epochs) + "  -  Validation Error: ", (1 - accuracy))

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, sess.run(global_step))
            saver.save(sess, "logs/model-checkpoint", global_step=global_step)

    print("Optimization Finished!")
    train_images = x_train[5000:]
    train_labels = y_train[5000:]
    test_feed_dict = {
        x: x_test,
        y: y_test
    }
    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
    print("Test Accuracy: ", accuracy)

