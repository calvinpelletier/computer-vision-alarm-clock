import os

IS_RPI = os.environ['IS_RASPBERRYPI'].lower() == 'true'
if IS_RPI:
    MEAN_STD_DEV_PATH = '/root/computer-vision-alarm-clock/mean_std_dev.csv'
    CROP_PATH = '/root/computer-vision-alarm-clock/crop.csv'
else:
    MEAN_STD_DEV_PATH = '/home/calvin/projects/computer-vision-alarm-clock/mean_std_dev.csv'
    CROP_PATH = '/home/calvin/projects/computer-vision-alarm-clock/crop.csv'

def save_mean_std_dev(mean, std_dev):
    with open(MEAN_STD_DEV_PATH, 'w') as f:
        f.write('{},{}\n'.format(mean, std_dev))
    print('saved mean = {}, std_dev = {}'.format(mean, std_dev))

def load_mean_std_dev():
    with open(MEAN_STD_DEV_PATH, 'r') as f:
        mean, std_dev = [float(x) for x in f.readlines()[0].strip().split(',')]
    print('loaded mean = {}, std_dev = {}'.format(mean, std_dev))
    return ((mean,), (std_dev,))

def get_crop():
    with open(MEAN_STD_DEV_PATH, 'r') as f:
        x1, x2, y1, y2 = [int(x) for x in f.read().strip().split(',')]
    return 30, 291, 45, 145
