import cv2
import numpy as np


def process_graph(image_dir):

    img = cv2.imread(image_dir)
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Finding Contours
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("Image", image)
    #cv2.waitKey(0)

    candidates = []
    invalid = []
    valid_cons = []

    # save list of all contours values
    for cnt in contours:
        if cv2.contourArea(cnt) > 45:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 10 & w < 25:
                candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'contour': cnt})
                # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # get list of contours found inside other contours
    # these are invalid and cannot be used
    for a in candidates:

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

    graph_connections = []
    graph_nodes = []
    # save contours that are valid
    for c1 in candidates:
        if c1 not in invalid:
            valid_cons.append(c1)
            hull = cv2.convexHull(c1['contour'], False)
            #cv2.drawContours(image, [hull], 0, (155, 155, 155), 2, 8)
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            circularity = (perimeter * perimeter) / (4 * np.pi * area)
            #print(circularity)

            if circularity <= 1.7:
                graph_nodes.append({'contour': c1, 'hull': hull})

            else:
                # list as connection, not node
                graph_connections.append({'contour': c1, 'hull': hull})

               # rows, cols = image.shape[:2]
               # [vx, vy, x, y] = cv2.fitLine(c1['contour'], cv2.DIST_L2, 0, 0.01, 0.01)
               # lefty = int((-x * vy / vx) + y)
               # righty = int(((cols - x) * vy / vx) + y)
               # cv2.line(image, (cols - 1, righty), (0, lefty), (100, 100, 100), 2)
               # cv2.imshow("Image", image)
               # cv2.waitKey(0)

    node_images = []
    for g_node in graph_nodes:
        # create mask to hide all other contours
        mask = np.ones(image.shape[:2], dtype="uint8") * 255

        for c in valid_cons:
            if c is not g_node:
                cv2.drawContours(mask, [c['contour']], -1, 0, -1)

        # remove the contours from the image and show the resulting images
        # masked_image = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Image", masked_image)
        # cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(g_node['hull'])
        node_image = image[y: y + h, x: x + w]
        node_images.append(node_image)
        #cv2.imshow("Image", node_image)
       # cv2.waitKey(0)

    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    return node_images


if __name__ == '__main__':
    # img_dir = "images/graphs/circle_test.png"
    # img_dir = 'images/graphs/graph_test.png'
    # img_dir = 'images/graphs/graph_test_2.png'
    img_dir = 'images/graphs/graph_test_3.png'

    imgs = process_graph(img_dir)
    print(imgs)
