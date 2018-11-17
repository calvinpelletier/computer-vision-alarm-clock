#!/usr/bin/python3
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from data_preprocessor import TrainDataPreprocessor, TorchDataset, load_mean_std_dev

np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)

EPOCHS = 15
IS_GPU = True
TOTAL_CLASSES = 2
TRAIN_BS = 32
VAL_BS = 32


def calculate_val_accuracy(valloader, is_gpu):
    """
    Args:
        valloader (torch.utils.data.DataLoader): val set
        is_gpu (bool): whether to run on GPU
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """
    correct = 0.
    total = 0.
    predictions = []

    class_correct = list(0. for i in range(TOTAL_CLASSES))
    class_total = list(0. for i in range(TOTAL_CLASSES))

    for data in valloader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)
        correct += (predicted == labels).sum()

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_accuracy = 100 * np.divide(class_correct, class_total)
    return 100*correct/total, class_accuracy


data = TrainDataPreprocessor(percent_training=0.99)

# mean and std dev per channel
train_means, train_stds = load_mean_std_dev()
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(train_means, train_stds)])
val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(train_means, train_stds)])

trainset = TorchDataset(data.x_train, data.y_train, train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BS,
                                          shuffle=True, num_workers=2)
print("Train set size: "+str(len(trainset)))

valset = TorchDataset(data.x_val, data.y_val, val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BS,
                                         shuffle=False, num_workers=2)
print("Val set size: "+str(len(valset)))


classes = ['out-of-bed', 'in-bed']

FITNESS_SCALE = 4. # how favorably to treat highly fit models
TARGET_MODELS_PER_GEN = 20
N_GENS = 1
BREAK_AFTER_X_EPOCHS_WITHOUT_MAX = 3
MAX_EPOCHS = 40
STRICT = True
if len(sys.argv) > 1:
    idx = int(sys.argv[1])
else:
    idx = 0
print('index: {}'.format(idx))
OUT_FILE = 'results{}.txt'.format(idx)

initial_populations = [
    [c5pc5pc5pfn(5, 10, 10, 100)],
    [c5pc5pc5pf(5, 10, 10, 100)],
    [c13pc13pf(6, 16, 50)],
    [c13pc13pfn(6, 16, 50)],
    [c13pc13pff(6, 16, 50, 20)],
    [c5pc5pc5pff(5, 10, 10, 100, 20)],
    [c5pc5pfff(5, 5, 300, 200, 100)],
]
population = initial_populations[idx]
results = {}
best_val_acc = 0.
for gen in range(N_GENS):
    print('###########GEN {}###########'.format(gen))
    for model in population:
        print('creating model...')
        net = model
        if net.label in results:
            net.accuracy = results[net.label]
            continue
        print('~~~~~{}~~~~~'.format(net.label))

        print('transfering data to gpu...')
        if IS_GPU:
            net = net.cuda()
        print('done')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

        # loop until there hasnt been a new max val accuracy for x epochs
        max_val_accuracy = 0.
        iters_since_max = 0
        epoch = 0
        while 1:
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                if IS_GPU:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

            # Normalizing the loss by the total number of train batches
            running_loss/=len(trainloader)
            print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))

            # Scale of 0.0 to 100.0
            # Calculate validation set accuracy of the existing model
            val_accuracy, val_classwise_accuracy = \
                calculate_val_accuracy(valloader, IS_GPU)
            print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

            # Optionally print classwise accuracies
            for c_i in range(TOTAL_CLASSES):
                print('Accuracy of %s : %2d %%' % (
                    classes[c_i], 100 * val_classwise_accuracy[c_i]))

            # if using while loop
            epoch += 1
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                iters_since_max = 0
            else:
                iters_since_max += 1
            if iters_since_max > BREAK_AFTER_X_EPOCHS_WITHOUT_MAX or epoch >= MAX_EPOCHS:
                EPOCHS = epoch
                break
        # --- end epoch loop ------
        net.accuracy = max_val_accuracy
        results[net.label] = max_val_accuracy

        # save results to a txt file
        with open(OUT_FILE, 'a') as f:
            f.write('{}: {}\n'.format(net.label, net.accuracy))

        if net.accuracy > best_val_acc:
            best_val_acc = net.accuracy
            torch.save(net.state_dict(), 'best_model.pt')

    # --- end population loop ----------
    population.sort(key=lambda x:x.accuracy)
    new_pop = []
    if STRICT:
        for _ in range(TARGET_MODELS_PER_GEN):
            new_pop.append(mutate(population[-1]))
    else:
        pop_size = len(population)
        std_devs = np.array(list(range(1,pop_size+1)), dtype=float) * FITNESS_SCALE
        target_sum_std_dev = float(TARGET_MODELS_PER_GEN) / np.sqrt(2. / np.pi) # because expected value of half-normal distribution is sigma*sqrt(2/pi)
        adj_std_devs = std_devs / (np.sum(std_devs) / target_sum_std_dev)
        model_types = {}
        for i in range(pop_size-1, -1, -1):
            n_children = int(np.round(np.abs(np.random.normal(scale=adj_std_devs[i]))))
            print('{} children: {}'.format(population[i].label, n_children))
            for _ in range(n_children):
                new_pop.append(mutate(population[i]))

            # let best of each model type go on to next gen without mutations
            # so that there's always at least one of each type
            model_type = population[i].label.split('(')[0]
            if model_type not in model_types:
                new_pop.append(population[i])
                model_types[model_type] = True
    population = new_pop


# --- end generation loop ---------
for label, acc in results.items():
    print(label + ': ' + str(acc))

# Plot train loss over epochs and val set accuracy over epochs
# plt.subplot(2, 1, 1)
# plt.ylabel('Train loss')
# plt.plot(np.arange(EPOCHS), train_loss_over_epochs, 'k-')
# plt.title('(NetID) train loss and val accuracy')
# plt.xticks(np.arange(EPOCHS, dtype=int))
# plt.grid(True)
#
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(EPOCHS), val_accuracy_over_epochs, 'b-')
# plt.ylabel('Val accuracy')
# plt.xlabel('Epochs')
# plt.xticks(np.arange(EPOCHS, dtype=int))
# plt.grid(True)
# plt.savefig("plot.png")
# plt.close(fig)
# print('Finished Training')
