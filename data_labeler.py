#!/usr/bin/python3
import os
import cv2

import constants as c
from util import get_crop

SQUARE_SIZE = 80
CROP_X1, CROP_X2, CROP_Y1, CROP_Y2 = get_crop()

count = 0
face_x, face_y = None, None

def mouse_callback(event, x, y, flags, param):
    global face_x, face_y, image
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

    if c.IS_HORIZONTAL_ALIGNMENT:
        subimage_h = h
        subimage_w = h
    else:
        subimage_h = w
        subimage_w = w

    face_file = open(c.FACE_PATH, 'a')
    non_face_file = open(c.NON_FACE_PATH, 'a')

    if c.IS_HORIZONTAL_ALIGNMENT:
        for x in range(0, w - subimage_w, 20):
            x1 = CROP_X1 + x
            x2 = x1 + subimage_w
            y1 = CROP_Y1
            y2 = CROP_Y2
            if face_x is not None and x + subimage_w > face_x and x < face_x:
                # at least half of face in picture
                # cv2.imwrite(os.path.join(FACE_PATH, '{}.png'.format(count)), og_image[y1:y2, x1:x2])
                # count += 1
                face_file.write('{},{},{},{},{}\n'.format(
                    filename, x1, x2, y1, y2))
            else:
                # face not in picture
                # cv2.imwrite(os.path.join(NON_FACE_PATH, '{}.png'.format(count)), og_image[y1:y2, x1:x2])
                # count += 1
                non_face_file.write('{},{},{},{},{}\n'.format(
                    filename, x1, x2, y1, y2))
    else:
        for y in range(0, h - subimage_h, 20):
            y1 = CROP_Y1 + y
            y2 = y1 + subimage_h
            x1 = CROP_X1
            x2 = CROP_X2
            if face_y is not None and y + subimage_h > face_y and y < face_y:
                # at least half of face in picture
                # cv2.imwrite(os.path.join(FACE_PATH, '{}.png'.format(count)), og_image[y1:y2, x1:x2])
                # count += 1
                face_file.write('{},{},{},{},{}\n'.format(
                    filename, x1, x2, y1, y2))
            else:
                # face not in picture
                # cv2.imwrite(os.path.join(NON_FACE_PATH, '{}.png'.format(count)), og_image[y1:y2, x1:x2])
                # count += 1
                non_face_file.write('{},{},{},{},{}\n'.format(
                    filename, x1, x2, y1, y2))

    print('done')


cv2.namedWindow('labeler')
cv2.setMouseCallback('labeler', mouse_callback)
images = sorted(
    os.listdir(c.UNLABELED_PATH),
    key=lambda x: int(x.rstrip('.jpeg').replace('-', ''))
)
print('press "s" to skip or "c" to create training data. click 1 location to mark face before pressing "c" if present.')
for filename in images:
    print(filename)
    filepath = os.path.join(c.UNLABELED_PATH, filename)
    og_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = og_image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    cv2.imshow('labeler', image)

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        # escape
        break
    elif k == ord('s'):
        os.rename(filepath, os.path.join(c.SKIPPED_PATH, filename))
        continue
    elif k == ord('c'):
        image_to_training_data(filename, image)
        os.rename(filepath, os.path.join(c.LABELED_PATH, filename))
        face_x, face_y = None, None
    else:
        print('invalid key press, skipping image anyway')
