#!/usr/bin/python3
import os
import cv2

from util import get_crop

VERSION = 'main1'
UNLABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/unlabeled'.format(VERSION)
FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/face.csv'.format(VERSION)
NON_FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/non-face.csv'.format(VERSION)
LABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/labeled'.format(VERSION)
SKIPPED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/skipped'.format(VERSION)
CROP_X1, CROP_X2, CROP_Y1, CROP_Y2 = get_crop()

count = 0
face_x, face_y = None, None

def mouse_callback(event, x, y, flags, param):
    global face_x, face_y, image
    SQUARE_SIZE = 80
    if event == cv2.EVENT_LBUTTONDOWN:
        print('click ({}, {})'.format(x, y))
        face_x = x
        face_y = y
        x1 = x - SQUARE_SIZE // 2
        y1 = y - SQUARE_SIZE // 2
        x2 = x + SQUARE_SIZE // 2
        y2 = y + SQUARE_SIZE // 2
        assert(x2 - x1 == SQUARE_SIZE and y2 - y1 == SQUARE_SIZE)
        image_with_rect = image.copy()
        cv2.rectangle(
            image_with_rect,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1)
        cv2.imshow('labeler', image_with_rect)


def image_to_training_data(filename, image):
    global face_x, face_y, count, og_image
    print('creating training data...')
    h, w = image.shape
    subimage_h = h
    subimage_w = h

    face_file = open(FACE_PATH, 'a')
    non_face_file = open(NON_FACE_PATH, 'a')

    for x in range(0, w - subimage_w, 20):
        x1 = CROP_X1 + x
        x2 = x1 + subimage_w
        y1 = CROP_Y1
        y2 = CROP_Y2
        if face_x is not None and x + subimage_w > face_x and x < face_x:
            # at least half of face in picture
            # cv2.imwrite(os.path.join(FACE_PATH, '{}.png'.format(count)), og_image[y1:y2, x1:x2])
            # count += 1
            face_file.write('{},{},{},{},{}\n'.format(filename, x1, x2, y1, y2))
        else:
            # face not in picture
            # cv2.imwrite(os.path.join(NON_FACE_PATH, '{}.png'.format(count)), og_image[y1:y2, x1:x2])
            # count += 1
            non_face_file.write('{},{},{},{},{}\n'.format(filename, x1, x2, y1, y2))

    print('done')


cv2.namedWindow('labeler')
cv2.setMouseCallback('labeler', mouse_callback)
images = sorted(os.listdir(UNLABELED_PATH), key=lambda x: int(x.rstrip('.jpeg').replace('-', '')))
print('press "s" to skip or "c" to create training data. click 2 locations to mark face before pressing "c" if present.')
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
        continue
    elif k == ord('c'):
        image_to_training_data(filename, image)
        os.rename(filepath, os.path.join(LABELED_PATH, filename))
        face_x, face_y = None, None
    else:
        print('invalid key press, skipping image anyway')
