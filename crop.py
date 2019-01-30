import sys
import cv2

from util import get_crop

CROP_X1, CROP_X2, CROP_Y1, CROP_Y2 = get_crop()
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
cv2.imwrite('test.png', img)
