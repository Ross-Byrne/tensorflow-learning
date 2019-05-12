import image_reader
import emnist_predictor
import graph_processor


if __name__ == '__main__':

    graph_image_dir = 'images/graphs/name-graph.png'
    # graph_image_dir = 'images/bench_04.png'
    # graph_image_dir = 'images/graphs/g-28.png'

    graph_nodes, graph_links = graph_processor.process_graph(graph_image_dir)

    nodes = []
    links = []

    # nodes found
    if len(graph_nodes):

        # build link structure
        for l in graph_links:
            if 'n1_index' in l and 'n2_index' in l:
                links.append({'source': l['n1_index'], 'target': l['n2_index']})

        # get text from nodes
        for node in graph_nodes:

            # read characters from image
            characters = image_reader.read_image(graph_image_dir, node)

            if len(characters) > 0:
                print('Classifying text for Node:', str(node['index'] + 1), "/", str(len(graph_nodes)), ". . .")

                text = ''
                # predict each character
                for char in characters:

                    if char is not None:
                        # Predict image, getting json as return type
                        prediction = emnist_predictor.predict(char)
                        text += prediction['prediction']
                        # print(prediction)
                    else:
                        text += ' '
                        # print('')
                # print(node)
                nodes.append({'index': node['index'], 'text': text})
            else:
                nodes.append({'index': node['index'], 'text': ''})

        # build json representation of graph
        graph_json = {'nodes': nodes, 'links': links}

        print('Nodes:', nodes)
        print('Links:', links)
        print('Graph JSON:', graph_json)

        # print out connected nodes
        for l in graph_json['links']:
            n1 = None
            n2 = None

            for n in graph_json['nodes']:
                if n['index'] is l['source']:
                    n1 = n

                if n['index'] is l['target']:
                    n2 = n

            if n1 is not None and n2 is not None:
                print(n1['text'] or n1['index'], "<-->", n2['text'] or n2['index'])

    else:
        print("No graph nodes detected...")
