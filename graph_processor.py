import cv2
import numpy as np


def calc_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x1 - x2
    dy = y1 - y2
    d = np.sqrt(dx * dx + dy * dy)

    return d


def process_graph(image_dir):

    img = cv2.imread(image_dir)

    dim = (1000, 500)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #cv2.imshow("Image", img)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Finding Contours
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

    candidates = []
    invalid = []
    valid_cons = []
    parent_child_list = []

    # save list of all contours values
    for cnt in contours:
        if cv2.contourArea(cnt) > 45:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 10 & w < 25:
                candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'contour': cnt})

    # get list of contours found inside other contours
    # these are invalid and cannot be used
    for a in candidates:
        parent_child_hash = {'parent': a, 'children': []}

        x1 = a['x']
        y1 = a['y']
        h1 = a['h']
        w1 = a['w']
        xw1 = x1 + w1
        yh1 = y1 + h1

        for b in candidates:
            if a != b:
                x2 = b['x']
                y2 = b['y']
                h2 = b['h']
                w2 = b['w']

                xw2 = x2 + w2
                yh2 = y2 + h2

                # check if contour B rect is inside of contour A rect
                # Yes, this took a few minutes to work out on paper...
                is_width = ((xw1 >= x2) and (x2 >= x1)) and ((xw1 >= xw2) and (xw2 >= x1))
                is_height = ((yh1 >= y2) and (y2 >= y1)) and ((yh1 >= yh2) and (yh2 >= y1))

                # if contour B is inside A, save it
                if is_width and is_height:
                    invalid.append(b)
                    parent_child_hash['children'].append(b)

        parent_child_list.append(parent_child_hash)

    graph_links = []
    graph_nodes = []
    n_index = 0
    # save contours that are valid
    for c1 in candidates:
        if c1 not in invalid:
            valid_cons.append(c1)
            hull = cv2.convexHull(c1['contour'], False)
            #cv2.drawContours(image, [hull], 0, (155, 155, 155), 2, 8)
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            circularity = (perimeter * perimeter) / (4 * np.pi * area)

            if circularity <= 1.7:
                c1['index'] = n_index
                graph_nodes.append(c1)
                n_index = n_index + 1

            else:
                # list as connection, not node
                graph_links.append(c1)

    # Find the nodes each link is pointing at
    for link in graph_links:
        l_cnt = link['contour']
        dis = []
        # get the two farthest points from each other, these are link ends
        for c1 in l_cnt:
            for c2 in l_cnt:
                d = calc_distance(c1[0], c2[0])
                if d > 0.0:
                    dis.append({'d': d, 'p1': c1, 'p2': c2})

        dis = sorted(dis, key=lambda z: z['d'])

        # extreme ends of link
        p1 = dis[len(dis)-1]['p1']
        p2 = dis[len(dis) - 1]['p2']

        p1_dis = []
        p2_dis = []

        # get the closest node to each end of the link
        for n in graph_nodes:
            n_cnt = n['contour']

            M = cv2.moments(n_cnt)
            nx = int(M['m10'] / M['m00'])
            ny = int(M['m01'] / M['m00'])

            # calc distance from each end
            p1d = calc_distance((nx, ny), p1[0])
            p2d = calc_distance((nx, ny), p2[0])

            if p1d > 0.0:
                p1_dis.append({'d': p1d, 'n': n})

            if p2d > 0.0:
                p2_dis.append({'d': p2d, 'n': n})

        # sort and get closest nodes to each end
        p1_dis = sorted(p1_dis, key=lambda z: z['d'])
        p2_dis = sorted(p2_dis, key=lambda z: z['d'])

        na = p1_dis[0]['n']
        nb = p2_dis[0]['n']

        # save node indices if not the same
        if na['index'] is not nb['index']:
            link['na_index'] = na['index']
            link['nb_index'] = nb['index']

    nodes_info = []
    for node in graph_nodes:
        for parent_child_hash in parent_child_list:

            if parent_child_hash['parent'] is node:
                nodes_info.append({'node': node, 'nested_contours': parent_child_hash['children']})

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    print('nodes:', str(len(nodes_info)), ' links:', str(len(graph_links)))

    return nodes_info, graph_links


if __name__ == '__main__':
    # img_dir = "images/graphs/circle_test.png"
    # img_dir = 'images/graphs/graph_test.png'
    # img_dir = 'images/graphs/graph_test_2.png'
    # img_dir = 'images/graphs/graph_test_3.png'
    img_dir = 'images/graphs/name-graph.png'

    nodes, links = process_graph(img_dir)
    #print('nodes:', str(len(nodes)), ' links:', str(len(links)))
