import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist as input_data
from scipy.io import loadmat

# Data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
mat_data_path = "emnist/emnist-balanced.mat"

# Parameters
learning_rate = 0.01
momentum = 0.5
training_epochs = 100
batch_size = 100
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
        output = layer(hidden_2, [256, 10], [10])

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

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    print(mnist)

    mat = loadmat(mat_data_path)
    #print(mat['dataset'][0][0][0][0][0])
    sess.run(init_op)

