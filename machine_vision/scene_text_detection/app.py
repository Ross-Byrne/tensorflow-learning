import sys
import numpy as np
import cv2

im = cv2.imread('../../images/scene_text_detection/bench_04.png')
# im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Now finding Contours

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
samples = np.empty((0, 100))
responses = []
candidates = []
invalid = []
valid_chars = []


# save list of all contours values
for cnt in contours:
    if cv2.contourArea(cnt) > 50:
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


# draw bounding boxes around valid characters
i = 0
for con in valid_chars:
    x = con['x']
    y = con['y']
    h = con['h']
    w = con['w']

    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
    roi = thresh[y: y + h, x: x + w]
    cv2.imwrite('images/img_' + str(i) + '.png', roi)  # save contents of rectangle to image folder
    cv2.imshow('norm', im)
    i += 1


cv2.waitKey(0)
sys.exit()
