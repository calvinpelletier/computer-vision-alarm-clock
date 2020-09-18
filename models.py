import torch.nn as nn
import torch.nn.functional as F
from random import randint

import constants as c

TOTAL_CLASSES = 2
INITIAL_CHANNELS = 1

def mutate(obj):
    new_params = []
    for param, t in zip(obj.params, obj.param_types):
        if t == 'c':
            new_params.append(randint(max(3, param-30), min(300, param+30)))
        elif t == 'f':
            new_params.append(randint(max(10, param-100), min(1200, param+100)))
    obj_type = obj.__class__
    return obj_type(*new_params)


# ~~~~~ 2 CONV LAYERS ~~~~~
class c7pc7pfff(nn.Module):
    def __init__(self, c1, c2, f1, f2, f3):
        super(c7pc7pfff, self).__init__()
        self.params = [c1, c2, f1, f2, f3]
        self.param_types = ['c', 'c', 'f', 'f', 'f']
        self.label = 'c7pc7pfff(c1={},c2={},f1={},f2={},f3={})'.format(*self.params)
        self.conv1 = nn.Conv2d(3, c1, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 7)
        dim = ((c.IMAGE_SIZE - 6) / 2 - 6) / 2
        if not dim.is_integer():
            raise Exception('invalid dim for c7pc7pfff')
        dim = int(dim)
        self.flat_size = c2 * dim * dim
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, f3),
            nn.ReLU(inplace=True),
            nn.Linear(f3, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x


class c11pc11pf(nn.Module):
    def __init__(self, c1, c2, f1):
        super(c11pc11pf, self).__init__()
        self.params = [c1, c2, f1]
        self.param_types = ['c', 'c', 'f']
        self.label = 'c11pc11pf(c1={},c2={},f={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 11)
        dim = ((c.IMAGE_SIZE - 10) / 2 - 10) / 2
        if not dim.is_integer():
            raise Exception('invalid dim for c11pc11pf: {}'.format(dim))
        dim = int(dim)
        self.flat_size = c2 * dim * dim
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x


class c11pc11pfn(nn.Module):
    def __init__(self, c1, c2, f1):
        super(c11pc11pfn, self).__init__()
        self.params = [c1, c2, f1]
        self.param_types = ['c', 'c', 'f']
        self.label = 'c11pc11pfn(c1={},c2={},f={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 11)
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)
        dim = ((c.IMAGE_SIZE - 10) / 2 - 10) / 2
        if not dim.is_integer():
            raise Exception('invalid dim for c11pc11pfn')
        dim = int(dim)
        self.flat_size = c2 * dim * dim
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x


class c11pc11pff(nn.Module):
    def __init__(self, c1, c2, f1, f2):
        super(c11pc11pff, self).__init__()
        self.params = [c1, c2, f1, f2]
        self.param_types = ['c', 'c', 'f', 'f']
        self.label = 'c11pc11pff(c1={},c2={},f1={},f2={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 11)
        dim = ((c.IMAGE_SIZE - 10) / 2 - 10) / 2
        if not dim.is_integer():
            raise Exception('invalid dim for c11pc11pff')
        dim = int(dim)
        self.flat_size = c2 * dim * dim
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~ 3 CONV LAYERS ~~~~~
class c5pc5pc5pf(nn.Module):
    def __init__(self, c1, c2, c3, f1):
        super(c5pc5pc5pf, self).__init__()
        self.params = [c1, c2, c3, f1]
        self.param_types = ['c', 'c', 'c', 'f']
        self.label = 'c5pc5pc5pf(c1={},c2={},c3={},f1={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.conv3 = nn.Conv2d(c2, c3, 5)
        self.flat_size = c3 * 9 * 9
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x


class c5pc5pc5pff(nn.Module):
    def __init__(self, c1, c2, c3, f1, f2):
        super(c5pc5pc5pff, self).__init__()
        self.params = [c1, c2, c3, f1, f2]
        self.param_types = ['c', 'c', 'c', 'f', 'f']
        self.label = 'c5pc5pc5pff(c1={},c2={},c3={},f1={},f2={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.conv3 = nn.Conv2d(c2, c3, 5)

        self.flat_size = c3 * 9 * 9
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x


class c5pc5pc5pfn(nn.Module):
    def __init__(self, c1, c2, c3, f1):
        super(c5pc5pc5pfn, self).__init__()
        self.params = [c1, c2, c3, f1]
        self.param_types = ['c', 'c', 'c', 'f']
        self.label = 'c5pc5pc5pfn(c1={},c2={},c3={},f1={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.conv3 = nn.Conv2d(c2, c3, 5)
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)
        self.norm3 = nn.BatchNorm2d(c3)

        self.flat_size = c3 * 9 * 9
        self.fc_net = nn.Sequential(
            nn.Linear(self.flat_size, f1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(f1),
            nn.Linear(f1, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = self.pool(self.norm3(F.relu(self.conv3(x))))
        x = x.view(-1, self.flat_size)
        x = self.fc_net(x)
        return x
