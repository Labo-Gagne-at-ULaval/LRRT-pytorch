# -*- coding: utf-8 -*-
"""
Summary : Learning rate range test (LRRT); fast version
Author : Stéphane M. Gagné, Université Laval
Version : 0.1-alpha

Description
===========
- Will perform a learning rate range test (LRRT) by gradually increasing the learning rate from a low value while
  following metric (train loss, train accuracy, test loss ans test accuracy).
- Will save the results in a .csv file
- This "fast" version continuously increase the LR within the same training run.
- This "fast" version is more noisy than the "slow" version.
- This "fast" version will have a different profile depending on parameters like lr_min. More sensitive to some
  parameters compared to the slow version.


Design
======
- Using this script should not require any modification directly in the script. Should use parameters_for_lrrt.py file.
- Should be independent of all model specific parameters : model, optimizer, loss. This should be defined in the input
  file.
- Input file should contain functions for model, optimizer, loss
- Args : input file, output basename, batch size, starting LR, final LR, fast mode, slow mode [TO DO]

Input file
==========
The input file should be in the current directory, and be specific for one particular test
The input file should have the following functions
- model
- optimizer
- loss
- dataloader
- test function
- parameters :  train_size, train_batch_size, epochs, step_size, lr_min, lr_max, momentum

Output
======
- results in a .csv file
- plots in a PDF file
- other results? LR having the best metrics? LR at 50% metrics?
"""

import numpy as np
from parameters_for_lrrt import *

lrrt_points = int(train_size * epochs / step_size)     # number of points between lr=lr_min and lr=lrr_max
lrrt_factor = lr_max / (lr_min ** (1 / (lrrt_points - 1)))       # next_LR = prev_LR * lrr_factor
filename_base = 'results/lrrt_fast_bs'+str(train_batch_size)+'_ss'+str(step_size)+'_e'+str(epochs)
train_set, val_set = load_data()
train_loader, train_loader_for_metrics, val_loader = dataloaders(train_set, val_set)
device = torch.device("cuda:0")     # Options are "cuda:0" or "cpu"
model = set_model()
model.to(device)
criterion, optimizer = set_loss_optimizer(model, lr_min, momentum)
lr = optimizer.param_groups[0]['lr']
log_running_loss = []
log_running_accu = []
log_train_loss = []
log_train_accu = []
log_val_loss = []
log_val_accu = []
log_lr = []
samples = []
points = []
running_loss = correct = total = p = 0
print('#  #_Samples  LR  Train_Running_Loss  Train_Loss  Val_Loss  Train_Running_Accuracy  Train_Accuracy  Val_Accuracy')
for e in range(epochs):
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # Training pass
        optimizer.zero_grad()
        output = model(images.cuda())
        loss = criterion(output, labels.cuda())
        # This is where the model learns by backpropagation
        loss.backward()
        # And optimizes its weights here
        optimizer.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += labels.cuda().size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
        if total == step_size:
            running_loss = running_loss/(step_size/train_batch_size)
            model.eval()
            if do_full_train_metrics:
                train_loss, train_accu = test(train_loader_for_metrics, model, criterion)
            else:
                train_loss = train_accu = 0
            if do_full_val_metrics:
                val_loss, val_accu = test(val_loader, model, criterion)
            else:
                val_loss = val_accu = 0
            model.train()

            points.append(p+1)
            samples.append((p+1)*step_size)
            log_lr.append(lr)
            log_running_loss.append(running_loss)
            log_running_accu.append(correct/total)
            log_train_loss.append(train_loss)
            log_train_accu.append(train_accu)
            log_val_loss.append(val_loss)
            log_val_accu.append(val_accu)
            print("{:4d}  {:6d}  {:.2E}  {:6.3f}  {:6.3f}  {:6.3f}  {:6.4f}  {:6.4f}  {:6.4f}    {}".format(
                p+1, (p+1)*step_size, lr, running_loss, train_loss, val_loss, correct/total, train_accu,
                val_accu, "=" * round(100*correct/total)))
            optimizer.param_groups[0]['lr'] *= lrrt_factor
            lr = optimizer.param_groups[0]['lr']
            p += 1
            running_loss = correct = total = 0


log_train_accu = np.array(log_train_accu)
log_val_accu = np.array(log_val_accu)
results = np.transpose(np.array((points, log_lr, log_running_loss, log_train_loss, log_val_loss,
                                 log_running_accu, log_train_accu, log_val_accu)))
np.savetxt(filename_base + '.csv', results, fmt='%4d  %.5E  %7.4f  %7.4f  %7.4f  %6.4f  %6.4f  %6.4f')
