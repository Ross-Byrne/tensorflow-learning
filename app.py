import image_reader
import emnist_predictor
import shutil

if __name__ == '__main__':

    # get image to read text from
    image_dir = 'images/bench_04.png'

    # read characters from image
    images_directory, character_dirs = image_reader.read_image(image_dir)
    print('Classifying text...')

    text = '\n'
    # predict each character
    for char_dir in character_dirs:

        # Predict image, getting json as return type
        prediction = emnist_predictor.predict(char_dir)
        text += prediction['prediction']
        print(prediction)

    print(text)

    # clean up images by deleting the directory
    shutil.rmtree(images_directory)  # Clean up
