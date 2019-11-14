#!/usr/bin/python3
import datetime
import os
import subprocess
import time
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import signal
import shutil
import light

from models import *
from data_preprocessor import TorchDataset, load_mean_std_dev
from util import get_crop

# NOTE: dont forget to append "dwc_otg.fiq_fsm_mask=0x3" to /boot/cmdline.txt (same line)
# https://www.raspberrypi.org/forums/viewtopic.php?f=28&t=197089

IS_RPI = os.environ['IS_RASPBERRYPI'].lower() == 'true'

VERSION = 'main2'

if IS_RPI:
    MODEL_PATH = '/root/computer-vision-alarm-clock/best_model.pt'
    IMG_PATH = '/root/cv-alarm-clock-data/live.jpeg'
    UNLABELED_PATH = '/root/cv-alarm-clock-data/{}/unlabeled'.format(VERSION)
    ALARM_PATH = '/root/computer-vision-alarm-clock/alarm.wav'
else:
    MODEL_PATH = '/home/calvin/projects/computer-vision-alarm-clock/best_model.pt'
    IMG_PATH = '/home/calvin/storage/cv-alarm-clock-data/live.jpeg'
    UNLABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/unlabeled'.format(VERSION)
    ALARM_PATH = '/home/calvin/projects/computer-vision-alarm-clock/alarm.wav'

LIVE_BS = 9
TOTAL_CLASSES = 2
INTERVAL = 3
CROP_X1, CROP_X2, CROP_Y1, CROP_Y2 = get_crop()
THRESHOLD = 1.5

ACTIVE_START = 750
ACTIVE_END = 815
ALWAYS_ACTIVE = False

SAVE_START = 730
SAVE_END = 1030
SAVE_EVERY_N_INTERVALS = 10 # 0 for no save
ALWAYS_SAVE = False

LIGHT_START = 740
LIGHT_END = 1030
ALWAYS_LIGHT = False

DEBUG = False

def live_data_preprocessor():
    x_total = []
    y_total = []

    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    if DEBUG:
        cv2.imwrite('test.png', img)
    h, w = img.shape
    subimage_h = h
    subimage_w = h

    for x1 in range(0, w - subimage_w, 20):
        x_total.append(img[:, x1:x1+subimage_w])

    x_live = np.array(x_total, dtype=np.uint8)
    y_live = np.zeros(x_live.shape, dtype=int)

    return x_live, y_live

model = c5pc5pc5pfn(c1=5,c2=10,c3=10,f1=100)
if IS_RPI:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
else:
    model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

train_means, train_stds = load_mean_std_dev()
live_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(train_means, train_stds)])

active = False
count = 0
light_on = False
while 1:
    cur_datetime = datetime.datetime.now()
    int_time = int(cur_datetime.strftime("%H%M"))

    if ALWAYS_LIGHT or (int_time >= LIGHT_START and int_time < LIGHT_END):
        if not light_on:
            if IS_RPI:
                light.on()
            else:
                subprocess.check_call('light on', shell=True)
            light_on = True
    else:
        if light_on:
            if IS_RPI:
                light.off()
            else:
                subprocess.check_call('light off', shell=True)
            light_on = False

    if ALWAYS_ACTIVE or (int_time >= ACTIVE_START and int_time < ACTIVE_END) \
                     or (int_time >= SAVE_START and int_time < SAVE_END):
        subprocess.check_call('fswebcam -r 320x240 --no-banner {}'.format(IMG_PATH), shell=True)
        if SAVE_EVERY_N_INTERVALS != 0 and (int_time >= SAVE_START and int_time < SAVE_END or ALWAYS_SAVE):
            if count % SAVE_EVERY_N_INTERVALS == 0:
                shutil.copyfile(
                    IMG_PATH,
                    os.path.join(
                        UNLABELED_PATH,
                        cur_datetime.strftime("%Y-%m-%d-%H-%M-%S") + '.jpeg'
                    )
                )
            count += 1

    if ALWAYS_ACTIVE or (int_time >= ACTIVE_START and int_time < ACTIVE_END):
        x_live, y_live = live_data_preprocessor()
        liveset = TorchDataset(x_live, y_live, live_transform)
        liveloader = torch.utils.data.DataLoader(liveset, batch_size=LIVE_BS, shuffle=False, num_workers=2)

        confidences = []
        for x, y in liveloader:
            output = model(Variable(x)).data
            for out_of_bed, in_bed in output:
                confidences.append(in_bed)
        confidence = max(confidences)
        print('{}: {}'.format(cur_datetime.strftime("%Y-%m-%d-%H-%M-%S"), confidence))

        if confidence > THRESHOLD and not active:
            print('launching alarm')
            active = True
            alarm_process = subprocess.Popen('mplayer {} -loop 0'.format(ALARM_PATH),
                stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        elif confidence < THRESHOLD and active:
            print('stopping alarm')
            active = False
            os.killpg(os.getpgid(alarm_process.pid), signal.SIGTERM)
    elif active:
        print('stopping alarm')
        active = False
        os.killpg(os.getpgid(alarm_process.pid), signal.SIGTERM)

    time.sleep(INTERVAL)
