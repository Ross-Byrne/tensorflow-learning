import cv2
import numpy as np
import graph_processor


def read_image(image_dir, graph_node):

    im = cv2.imread(image_dir)
    dim = (1000, 500)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    candidates = []
    invalid = []
    valid_chars = []

    if graph_node is not None:
        node = graph_node['node']
        nested_contours = graph_node['nested_contours']
        parent_area = cv2.contourArea(node['contour'])

        for con in nested_contours:
            child_area = cv2.contourArea(con['contour'])

            # keep contour if area is at least 20% smaller then parent area
            if child_area < (parent_area * 0.8):
                candidates.append(con)
    else:
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 45:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 10 & w < 25:
                    candidates.append({'x': x, 'y': y, 'w': w, 'h': h})

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

    # save contours that are valid
    for c1 in candidates:
        if c1 not in invalid:
            valid_chars.append(c1)

    # order characters in correct order, from left to right
    valid_chars = sorted(valid_chars, key=lambda x_coords: x_coords['x'])
    imgs = []

    if len(valid_chars) > 0:

        # calculate median distance between characters to guess where spaces are
        spaces = []
        valid_chars[0]['distance_from_last'] = 0
        for x in range(1, len(valid_chars)):
            distance = valid_chars[x]['x'] - (valid_chars[x - 1]['x'] + valid_chars[x - 1]['w'])
            spaces.append(distance)
            valid_chars[x]['distance_from_last'] = distance

        median_space = np.median(spaces)
        min_space_size = median_space + (median_space * 1.6)  # guess what the smallest space width is

        # draw bounding boxes around valid characters
        for con in valid_chars:
            x = con['x']
            y = con['y']
            h = con['h']
            w = con['w']

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            img = thresh[y: y + h, x: x + w]
            img = cv2.bitwise_not(img)  # invert to get white background and black text

            # pad width or height to make image a square and add extra padding to help classification
            height, width = img.shape[:2]
            extra_pad = 6
            half_pad = int(extra_pad / 2)

            if width % 2 != 0:
                width += 1

            if height % 2 != 0:
                height += 1

            if width < height:
                border_w = int((height - width + extra_pad) / 2)
                border_h = half_pad
            elif width > height:
                border_h = int((width - height + extra_pad) / 2)
                border_w = half_pad
            else:
                border_h = half_pad
                border_w = half_pad

            img = cv2.copyMakeBorder(img, top=border_h, bottom=border_h, left=border_w, right=border_w,
                                     borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

            # Add None to indicate a space
            if con['distance_from_last'] > min_space_size:
                imgs.append(None)

            imgs.append(img)

    # cv2.imshow('norm', im)
    # cv2.waitKey(0)

    return imgs


if __name__ == '__main__':
    img_dir = 'images/graphs/name-graph.png'

    nodes, links = graph_processor.process_graph(img_dir)
    print('nodes:', str(len(nodes)), ' links:', str(len(links)))

    characters = read_image(img_dir, nodes[4])
    print("Characters:", str(len(characters)))
