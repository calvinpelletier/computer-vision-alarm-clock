import torch.nn as nn
import torch.nn.functional as F
from random import randint

TOTAL_CLASSES = 2
INITIAL_CHANNELS = 1
IMAGE_SIZE = 100

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
class c5pc5pfff(nn.Module):
    def __init__(self, c1, c2, f1, f2, f3):
        super(c5pc5pfff, self).__init__()
        self.params = [c1, c2, f1, f2, f3]
        self.param_types = ['c', 'c', 'f', 'f', 'f']
        self.label = 'c5pc5pfff(c1={},c2={},f1={},f2={},f3={})'.format(*self.params)
        self.conv1 = nn.Conv2d(3, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.flat_size = c2 * 22 * 22
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


class c13pc13pf(nn.Module):
    def __init__(self, c1, c2, f1):
        super(c13pc13pf, self).__init__()
        self.params = [c1, c2, f1]
        self.param_types = ['c', 'c', 'f']
        self.label = 'c13pc13pf(c1={},c2={},f={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 13)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 13)
        self.flat_size = c2 * 16 * 16
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


class c13pc13pfn(nn.Module):
    def __init__(self, c1, c2, f1):
        super(c13pc13pfn, self).__init__()
        self.params = [c1, c2, f1]
        self.param_types = ['c', 'c', 'f']
        self.label = 'c13pc13pfn(c1={},c2={},f={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 13)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 13)
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)
        self.flat_size = c2 * 16 * 16
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


class c13pc13pff(nn.Module):
    def __init__(self, c1, c2, f1, f2):
        super(c13pc13pff, self).__init__()
        self.params = [c1, c2, f1, f2]
        self.param_types = ['c', 'c', 'f', 'f']
        self.label = 'c13pc13pff(c1={},c2={},f1={},f2={})'.format(*self.params)
        self.conv1 = nn.Conv2d(INITIAL_CHANNELS, c1, 13)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 13)
        self.flat_size = c2 * 16 * 16
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
