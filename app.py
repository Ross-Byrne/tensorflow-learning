import image_reader
import emnist_predictor
import shutil

if __name__ == '__main__':

    # get image to read text from
    image_dir = 'images/bench_11.png'

    # read characters from image
    characters = image_reader.read_image(image_dir)
    print('Classifying text...')

    text = '\n'
    # predict each character
    for char in characters:

        if char is not None:
            # Predict image, getting json as return type
            prediction = emnist_predictor.predict(char)
            text += prediction['prediction']
            print(prediction)
        else:
            text += ' '
            print('')

    print(text)

    # clean up images by deleting the directory
    #shutil.rmtree(images_directory)  # Clean up
