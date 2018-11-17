#!/usr/bin/python3
import datetime
import os
import subprocess
import time

UNLABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/test3/unlabeled'
INTERVAL = 2
ACTIVE_START = 7 # hour of day
ACTIVE_END = 10
DEBUG = False

while 1:
    cur_datetime = datetime.datetime.now()
    if DEBUG or (cur_datetime.hour >= ACTIVE_START and cur_datetime.hour < ACTIVE_END):
        filepath = os.path.join(UNLABELED_PATH, cur_datetime.strftime("%Y-%m-%d-%H-%M-%S") + '.jpeg')
        print(filepath)
        subprocess.check_call('streamer -f jpeg -o {}'.format(filepath), shell=True)
        # subprocess.call('paplay /usr/share/sounds/freedesktop/stereo/audio-volume-change.oga', shell=True)
    time.sleep(INTERVAL)
