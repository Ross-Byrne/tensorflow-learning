# Code adapted from the following GitHub repository:
# https://github.com/Coopss/EMNIST/blob/master/server.py
import os
import cv2 as cv
from keras.models import model_from_yaml
import numpy as np
import pickle

# Mute tensorflow debugging information console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global parameters
image_width = 28
image_height = 28


# Take optional bin directory location and loads
# a previously trained model and it's weights.
# Returns the model
def load_model(bin_dir='bin/'):

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model


# Takes optional bin directory location and loads label mapping
# Returns the loaded label mapping object
def load_mapping(bin_dir='bin/'):
    mapping = pickle.load(open('%s/mapping.p' % bin_dir, 'rb'))
    return mapping


# Takes the model, label mapping and directory of image to predict
# Returns the prediction and confidence
# Character images must have white characters and black backgrounds
def predict(image):

    # Load trained model and weights from file
    model = load_model()

    # load label mappings
    mapping = load_mapping()

    # Resize image
    x = cv.resize(image, (image_width, image_height), interpolation=cv.INTER_AREA)

    # reshape image data for use in neural network
    x = x.reshape(1, image_width, image_height, 1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)

    # Format result
    result = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
              'confidence': str(max(out[0]) * 100)[:6]}

    return result


if __name__ == '__main__':

    # get image to predict
    image_dir = 'images/img_h_01.png'
    im = cv.imread(image_dir)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # convert image to black and white
    im = np.invert(im)  # convert to black background and white text
    # cv.imshow("Image", im)
    # cv.waitKey(0)

    # Predict image, getting json as return type
    prediction = predict(im)
    print(prediction)
