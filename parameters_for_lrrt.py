# -*- coding: utf-8 -*-
"""
Summary :   Configuration file for Learning rate range test (LRRT). This file should be specific for each test run.
            It serves as the input for lrrt_fast.py or lrrt_slow.py
Author : Author : Stéphane M. Gagné, Université Laval

Required content
================
Parameters : see section "REQUIRED PARAMETERS THAT AFFECT THE TEST" below
Functions (adjust for specific model, dataset, ot training approach)
    load_data()
    set_model()
    dataloaders(train_set, val_set)
    set_loss_optimizer(model, lr_0, momentum)
    test(loader, model, criterion)
"""
import torch
from torch import nn
import torchvision

# REQUIRED PARAMETERS THAT AFFECT THE TEST
# ==================================================================================================
train_size = 60000                      # number of samples in train dataset
train_batch_size = 100                  # mini-batch size
step_size = train_batch_size * 3        # increase LR after (step_size) samples, i.e train_batch_size * n
epochs = 2                              # LRRT length (in #epochs)
lr_min = 1E-6                           # initial LR
lr_max = 1                              # stop LRRT if lr > lr_max
momentum = 0.9
do_full_train_metrics = True           # After each step_size, compute metrics for full train dataset
do_full_val_metrics = True             # After each step_size, compute metrics for full val dataset
torch.manual_seed(42)                   # reproducibility
# ==================================================================================================

# Parameters specific to your model / dataset. May vary depending on model and/or dataset
train_batch_size_metrics = 1000         # Batch size for full metrics of train datasets
val_batch_size = 1000
train_num_workers = 3                   # should be (#cores - 1) for batch_size=1000, max 5 for batch_size=100
train_num_workers_metrics = 3           # should be (#cores - 1) for batch_size=1000
val_num_workers = 3                     # should be (#cores - 1) for batch_size=1000


def load_data():
    # two options for Normalize : ((0.5,), (0.5,)) or  ((0.1307,), (0.3081,))
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,)),
         ])
    train_set = torchvision.datasets.MNIST('/home/smg/projects/pytorch/mnist-tests/data',
                                           download=True, train=True, transform=transform)
    val_set = torchvision.datasets.MNIST('/home/smg/projects/pytorch/mnist-tests/data',
                                         download=True, train=False, transform=transform)
    return train_set, val_set


def set_model():
    hidden1_size = 10
    hidden2_size = 10
    model = nn.Sequential(nn.Linear(28*28, hidden1_size),
                          nn.ReLU(),
                          nn.Linear(hidden1_size, hidden2_size),
                          nn.ReLU(),
                          nn.Linear(hidden2_size, 10),
                          nn.LogSoftmax(dim=1)
                          )
    return model


def dataloaders(train_set, val_set):
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=train_batch_size,
                                               num_workers=train_num_workers,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)
    train_loader_for_metrics = torch.utils.data.DataLoader(train_set,
                                                           batch_size=train_batch_size_metrics,
                                                           num_workers=train_num_workers_metrics,
                                                           shuffle=False,
                                                           pin_memory=True,
                                                           drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=val_batch_size,
                                             num_workers=val_num_workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)
    iterations = len(train_loader)
    return train_loader, train_loader_for_metrics, val_loader


def set_loss_optimizer(model, lr_0, momentum):
    criterion = nn.NLLLoss()  # with NLLLoss, must close model with nn.LogSoftmax(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_0, momentum=momentum)
    return criterion, optimizer


def test(loader, model, criterion):
    # evaluation of model against whole dataset (training, validation or test dataset)
    correct = total = running_loss = 0
    with torch.no_grad():                               # disable gradients
        for images, labels in loader:                   # loop over all mini-batches
            img = images.view(len(labels), 784)         # flatten image
            output = model(img.cuda())                  # inference
            loss = criterion(output, labels.cuda())     # loss calculation
            running_loss += loss.item()                 # add mini-batch loss to full set loss
            _, predicted = output.max(1)                # extract the predicted labels
            total += labels.size(0)                     # count the total number of data points (images)
            correct += predicted.eq(labels.cuda()).sum().item()     # count how many labels are correct
        test_loss = running_loss / len(loader)          # divide the loss sum by the number of mini-batches
        test_accu = correct / total                   # calculate global accuracy
    return test_loss, test_accu
