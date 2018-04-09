# Code adapted from the following GitHub repository:
# https://github.com/Coopss/EMNIST/blob/master/training.py
# Used strictly for Research and Learning
import os
import pickle
import numpy as np
from scipy.io import loadmat

# Mute tensorflow debugging information console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Emnist():

    def load_data(mat_file_path='EMNIST/matlab/emnist-balanced.mat', width=28, height=28, max_=None, verbose=True):

        # Local functions
        def rotate(img):
            # Used to rotate images (for some reason they are transposed on read-in)
            flipped = np.fliplr(img)
            return np.rot90(flipped)

        # make sure bin directory exists
        bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
        if not os.path.exists(bin_dir):
            print('Creating bin/ directory...')
            os.makedirs(bin_dir)

        # Load convoluted list structure form loadmat
        mat = loadmat(mat_file_path)

        # Load char mapping
        mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
        pickle.dump(mapping, open('bin/mapping.p', 'wb'))

        # Load training data
        if max_ is None:
            max_ = len(mat['dataset'][0][0][0][0][0][0])
        training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
        training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

        # Load testing data
        if max_ is None:
            max_ = len(mat['dataset'][0][0][1][0][0][0])
        else:
            max_ = int(max_ / 6)

        testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
        testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

        # Reshape training data to be valid
        if verbose:
            _len = len(training_images)

        for i in range(len(training_images)):
            if verbose:
                print('Processing Training Images: %d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')

            training_images[i] = rotate(training_images[i])
        if verbose:
            print('')

        # Reshape testing data to be valid
        if verbose:
            _len = len(testing_images)

        for i in range(len(testing_images)):
            if verbose:
                print('Processing Testing Images: %d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')

            testing_images[i] = rotate(testing_images[i])

        if verbose:
            print('')

        # Convert type to float32
        training_images = training_images.astype('float32')
        testing_images = testing_images.astype('float32')

        # Normalize to prevent issues with model
        training_images /= 255
        testing_images /= 255

        nb_classes = len(mapping)

        return (training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes
