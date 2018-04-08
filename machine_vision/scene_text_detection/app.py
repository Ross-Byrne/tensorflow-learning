import sys
import numpy as np
import cv2

im = cv2.imread('../../images/scene_text_detection/bench_04.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Now finding Contours

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100))
responses = []
keys = [i for i in range(48, 58)]

i = 0
for cnt in contours:

    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        print(h)
        if h > 10 & w < 25:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y: y + h, x: x + w]
            cv2.imwrite('images/img_' + str(i) + '.png', roi)  # save contents of rectangle to image folder
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm', im)
    i = i + 1

cv2.waitKey(0)
sys.exit()

