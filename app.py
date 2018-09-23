import image_reader
import emnist_predictor
import graph_processor
import shutil
import cv2
import numpy as np

if __name__ == '__main__':

    # get image to read text from
    #image_dir = 'images/bench_11.png'

    graph_image_dir = 'images/graphs/name-graph.jpg'
    #graph_image_dir = 'images/bench_04.png'

    nodes, links = graph_processor.process_graph(graph_image_dir)

    for node in nodes:

        # read characters from image
        characters = image_reader.read_image(graph_image_dir, node)

        if len(characters) > 0:
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
