import sys
import cv2

from util import set_crop

x1, x2, y1, y2 = None, None, None, None
cropped = None

def mouse_callback(event, x, y, flags, param):
    global x1, x2, y1, y2, cropped
    if event == cv2.EVENT_LBUTTONDOWN:
        print('click ({}, {})'.format(x, y))
        if x1 is None:
            x1 = x
            y1 = y
            print('click bottom right crop location')
        else:
            x2 = x
            y2 = y
            cropped = img[y1:y2, x1:x2]
            print('press any key to reveal cropped image')

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('crop')
cv2.setMouseCallback('crop', mouse_callback)
cv2.imshow('crop', img)
print('click top left crop location')
k = cv2.waitKey(0) & 0xFF
cv2.imshow('crop', cropped)
print('press any key to save crop locations')
k = cv2.waitKey(0) & 0xFF
set_crop(x1, x2, y1, y2)
