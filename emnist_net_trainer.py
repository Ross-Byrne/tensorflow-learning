# Mute tensorflow debugging information console
import os
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
from emnist import Emnist  # Load EMNIST Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 256
epochs = 1


def build_net(training_data, width=28, height=28, verbose=False):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 64  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    model = Sequential()
    model.add(Conv2D(64, kernel_size, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # load weights into new model
    if os.path.isfile('bin/model.h5'):
        print('Loading model weights from file...')
        model.load_weights('bin/model.h5')

    if verbose is True:
        print(model.summary())

    return model


def train(model, training_data, callback=True, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if callback is True:
        # Callback for analysis in TensorBoard
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack] if callback else None)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/model.yaml", "w") as yaml_file:
        print('Saving model to bin/model.yaml...')
        yaml_file.write(model_yaml)

    print('Saving model weights to bin/model.h5...')
    save_model(model, 'bin/model.h5')


if __name__ == '__main__':

    training_data = Emnist.load_data()
    model = build_net(training_data)
    train(model, training_data, batch_size=batch_size, epochs=epochs)
