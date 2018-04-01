import tensorflow as tf
import numpy
import emnist as dataset
import EmnistData as emnistData

print("Testing EMNIST Data import")
batch_size = 100

data = emnistData.load_data()

