#!/usr/bin/python3
import os
import cv2

VERSION = 'test1'
UNLABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/unlabeled/'.format(VERSION)
FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/face/'.format(VERSION)
NON_FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/non-face/'.format(VERSION)
SKIPPED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/skipped/'.format(VERSION)
CROP_Y1 = 45
CROP_Y2 = 145
CROP_X1 = 30
CROP_X2 = 291

cv2.namedWindow('labeler')
images = sorted(os.listdir(UNLABELED_PATH), key=lambda x: int(x.rstrip('.jpeg').replace('-', '')))
print('press "f" for face or "n" for non face or "s" to skip')
for filename in images:
    print(filename)
    filepath = os.path.join(UNLABELED_PATH, filename)
    og_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = og_image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    cv2.imshow('labeler', image)

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        # escape
        break
    elif k == ord('s'):
        os.rename(filepath, os.path.join(SKIPPED_PATH, filename))
    elif k == ord('n'):
        os.rename(filepath, os.path.join(NON_FACE_PATH, filename))
    elif k == ord('f'):
        os.rename(filepath, os.path.join(FACE_PATH, filename))
    else:
        print('invalid key press, skipping image anyway')
