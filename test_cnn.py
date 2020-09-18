#!/usr/bin/python3
import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import *
from data_preprocessor import TorchDataset, load_mean_std_dev
from util import get_crop

MODEL_PATH = '/home/calvin/projects/computer-vision-alarm-clock/best_model.pt'
# TEST_FACE_FOLDER = '/home/calvin/storage/cv-alarm-clock-data/test2/face/'
# TEST_NON_FACE_FOLDER = '/home/calvin/storage/cv-alarm-clock-data/test2/non-face/'

LIVE_BS = 9
TOTAL_CLASSES = 2
CROP_X1, CROP_X2, CROP_Y1, CROP_Y2 = get_crop()

def live_data_preprocessor(img_path):
    x_total = []
    y_total = []

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    h, w = img.shape
    subimage_h = h
    subimage_w = h

    for x1 in range(0, w - subimage_w, 20):
        x_total.append(img[:, x1:x1+subimage_w])

    x_live = np.array(x_total, dtype=np.uint8)
    y_live = np.zeros(x_live.shape, dtype=int)

    return x_live, y_live


# https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744/2
model = c11pc11pfn(c1=30,c2=11,f1=87)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

train_means, train_stds = load_mean_std_dev()
live_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(train_means, train_stds)])

face_confidences = []
print('FACE')
for filename in os.listdir(TEST_FACE_FOLDER):
    x_live, y_live = live_data_preprocessor(TEST_FACE_FOLDER + filename)
    liveset = TorchDataset(x_live, y_live, live_transform)
    liveloader = torch.utils.data.DataLoader(liveset, batch_size=LIVE_BS, shuffle=False, num_workers=2)
    confidences = []
    for x, y in liveloader:
        output = model(Variable(x)).data
        for out_of_bed, in_bed in output:
            confidences.append(in_bed)
    confidence = max(confidences)
    face_confidences.append(confidence)
    print('{}: {}'.format(filename, confidence))

non_face_confidences = []
print('NON FACE')
for filename in os.listdir(TEST_NON_FACE_FOLDER):
    x_live, y_live = live_data_preprocessor(TEST_NON_FACE_FOLDER + filename)
    liveset = TorchDataset(x_live, y_live, live_transform)
    liveloader = torch.utils.data.DataLoader(liveset, batch_size=LIVE_BS, shuffle=False, num_workers=2)
    confidences = []
    for x, y in liveloader:
        output = model(Variable(x)).data
        for out_of_bed, in_bed in output:
            confidences.append(in_bed)
    confidence = max(confidences)
    non_face_confidences.append(confidence)
    print('{}: {}'.format(filename, confidence))

print('max non face = {}, min face = {}'.format(max(non_face_confidences), min(face_confidences)))

plt.scatter(
    face_confidences + non_face_confidences,
    np.random.rand(len(face_confidences) + len(non_face_confidences)),
    c=['red'] * len(face_confidences) + ['blue'] * len(non_face_confidences)
)
plt.savefig('test_results.png')
