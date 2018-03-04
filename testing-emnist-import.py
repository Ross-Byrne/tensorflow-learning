import tensorflow as tf
import numpy
import emnist as dataset

print("Testing EMNIST Data import")

data_dir = "emnist/emnist-binary/"
batch_size = 100


# Load the datasets
train_ds = dataset.train(data_dir).shuffle(60000).batch(batch_size)
test_ds = dataset.test(data_dir).batch(batch_size)
print(train_ds)
