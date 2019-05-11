import cv2 as cv
import numpy as np


def calc_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x1 - x2
    dy = y1 - y2
    d = np.sqrt(dx * dx + dy * dy)
    return d


def process_graph(image_dir):

    img = cv.imread(image_dir)
    (h, w) = img.shape[:2]
    r = 900 / float(w)
    dim = (900, int(h * r))
    # dim = (1000, 500)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # cv.imshow("img", img)
    # cv.imshow("gray", gray)
    # cv.imshow("blur", blur)
    cv.imshow("thresh", thresh)

    # Finding Contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    image = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv.imshow("Image", image)
    cv.waitKey(0)

    parent_child_list = []
    child_hash = {}
    invalid = []
    candidates = []

    # Build parent - child tree hash
    for cnt_index in range(len(contours)):
        cnt = contours[cnt_index]

        if cv.contourArea(cnt) <= 40:
            continue  # Contour too small, ignore

        parent = {'i': cnt_index, 'contour': cnt}
        parent_child = {'parent': parent, 'children': []}

        for child_index in range(len(contours)):
            child_cnt = contours[child_index]
            parent_index = hierarchy[0][child_index][3]  # the index of the parent

            if cnt_index == parent_index and cv.contourArea(child_cnt) > 40:
                child = {'i': child_index, 'contour': child_cnt}
                invalid.append(child)
                parent_child['children'].append(child)

        # Save parent and its children
        parent_child_list.append(parent_child)
        child_hash[cnt_index] = parent_child['children']

    # compile list of valid candidates
    for candidate in parent_child_list:
        if candidate['parent'] not in invalid:
            kids = [candidate['children']]

            # get second level children
            for child in candidate['children']:
                i = child['i']
                kids.append(child_hash[i])

            all_kids = [item for sublist in kids for item in sublist]  # flatten list of lists
            candidate['children'] = all_kids
            candidates.append(candidate)

    # # draw parent contour rectangles
    # for cnt in parent_child_list:
    #     if cnt['parent'] not in invalid:
    #         c = cnt['parent']
    #         [x, y, w, h] = cv.boundingRect(c['contour'])
    #         filtered = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # This is for demonstration
    #
    # cv.imshow("filtered", filtered)
    # cv.waitKey(0)

    graph_links = []
    graph_nodes = []
    n_index = 0

    # classify contours are nodes or edges
    for candidate in candidates:
        cnt = candidate['parent']
        area = cv.contourArea(cnt['contour'])
        perimeter = cv.arcLength(cnt['contour'], True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if circularity >= 0.7:
            cnt['index'] = n_index
            graph_nodes.append(cnt)
            n_index = n_index + 1
        else:
            graph_links.append(cnt)  # list as connection, not node

    # Find the nodes each link is pointing at, if there are any nodes
    if len(graph_nodes) > 1:
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
            p1 = dis[len(dis) - 1]['p1']
            p2 = dis[len(dis) - 1]['p2']

            p1_dis = []
            p2_dis = []

            # get the closest node to each end of the link
            for n in graph_nodes:
                n_cnt = n['contour']

                M = cv.moments(n_cnt)
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

            n1 = p1_dis[0]['n']
            n2 = p2_dis[0]['n']

            # save node indices if not the same
            if n1['index'] is not n2['index']:
                link['n1_index'] = n1['index']
                link['n2_index'] = n2['index']

    nodes_info = []
    for node in graph_nodes:
        for parent_child_hash in parent_child_list:

            if parent_child_hash['parent'] is node:
                nodes_info.append({'node': node, 'index': node['index'], 'nested_contours': parent_child_hash['children']})

    # cv.imshow("Image", image)
    # cv.waitKey(0)

    print('nodes:', len(nodes_info), ' links:', len(graph_links))
    return nodes_info, graph_links


if __name__ == '__main__':
    # img_dir = "images/graphs/circle_test.png"
    # img_dir = 'images/graphs/graph_test.png'
    # img_dir = 'images/graphs/graph_test_2.png'
    # img_dir = 'images/graphs/graph_test_3.png'
    # img_dir = 'images/graphs/name-graph.png'
    img_dir = 'images/graphs/g-13.png'

    nodes, links = process_graph(img_dir)
    # print('nodes:', str(len(nodes)), ' links:', str(len(links)))
