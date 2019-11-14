#!/usr/bin/python3
import os

IS_RPI = os.environ['IS_RASPBERRYPI'].lower() == 'true'
if IS_RPI:
    ROOT_CODE_PATH = '/root/computer-vision-alarm-clock'
    ROOT_DATA_PATH = '/root/cv-alarm-clock-data'
else:
    ROOT_CODE_PATH = '/home/calvin/projects/computer-vision-alarm-clock'
    ROOT_DATA_PATH = '/home/calvin/storage/cv-alarm-clock-data'

IS_HORIZONTAL_ALIGNMENT = False # interlaken used horizontal, hamlin used vertical
VERSION = 'main2'

UNLABELED_PATH = '{}/{}/unlabeled'.format(ROOT_DATA_PATH, VERSION)
FACE_PATH = '{}/{}/face.csv'.format(ROOT_DATA_PATH, VERSION)
NON_FACE_PATH = '{}/{}/non-face.csv'.format(ROOT_DATA_PATH, VERSION)
LABELED_PATH = '{}/{}/labeled'.format(ROOT_DATA_PATH, VERSION)
SKIPPED_PATH = '{}/{}/skipped'.format(ROOT_DATA_PATH, VERSION)
IMG_PATH = '{}/live.jpeg'.format(ROOT_DATA_PATH)
MODEL_PATH = '{}/best_model.pt'.format(ROOT_CODE_PATH)
ALARM_PATH = '{}/alarm.wav'.format(ROOT_CODE_PATH)
