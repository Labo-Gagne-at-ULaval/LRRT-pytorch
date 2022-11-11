# -*- coding: utf-8 -*-
"""
Summary : Learning rate range test (LRRT); slow version
Author: Stéphane M. Gagné, Université Laval
Version : 0.1-alpha

=====
USAGE
=====
- Must have the following file in current directory : parameters_for_lrrt.py
- Set parameters, model and training details in parameters_for_lrrt.py
- python lrrt_slow.py

Description
===========
- Will perform a learning rate range test (LRRT) by gradually increasing the learning rate from a low value while
  following metric (train loss, train accuracy, test loss ans test accuracy).
- Will save the results in a .csv file
- This "slow" version resets all model parameters before each LR increase. So every LR is tested from scratch.
- This "slow" version is less noisy than the "fast" version.
- This "slow" version provides a much cleaner LRRT profile.


Design
======
- Using this script should not require any modification directly in the script. Should use Args.
- Should be independent of all model specific parameters : model, optimizer, loss. This should be defined in the input
  file.
- Input file should contain functions for model, optimizer, loss
- Args : input file, output basename, batch size, starting LR, final LR, fast mode, slow mode [TO DO]

Input file
==========
filename : parameters_for_lrrt
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
from parameters_for_lrrt import *       # parameters_for_lrrt.py must be in current directory

filename_base = 'results/lrrt_slow_bs'+str(train_batch_size)+'_e'+str(epochs)+'_m'+str('{:.2f}'.format(momentum))
train_set, val_set = load_data()
train_loader, train_loader_for_metrics, val_loader = dataloaders(train_set, val_set)
model = set_model()
model.to(device)
criterion, optimizer = set_loss_optimizer(model, lr_min, momentum)
lr = optimizer.param_groups[0]['lr']
if do_full_train_metrics:
    log_train_loss = []
    log_train_accu = []
if do_full_val_metrics:
    log_val_loss = []
    log_val_accu = []
log_lr = []
points = []
print('                 Train     Val   Train     Val')
print('   #      LR      Loss    Loss    Accu    Accu')
print(' ===  ========   =====   =====   =====   =====')
torch.save(model, filename_base+'_model0.pt')
for p in range(lrrt_points):
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
            _, predicted = output.max(1)
    points.append(p+1)
    log_lr.append(lr)
    model.eval()
    train_loss, train_accu = test(train_loader_for_metrics, model, criterion)
    val_loss, val_accu = test(val_loader, model, criterion)
    model.train()
    log_train_loss.append(train_loss)
    log_train_accu.append(train_accu)
    log_val_loss.append(val_loss)
    log_val_accu.append(val_accu)
    print("{:4d}  {:.2E}  {:6.3f}  {:6.3f}  {:6.2f}  {:6.2f}    {}".format(
        p+1, lr, train_loss, val_loss, train_accu,
        val_accu, "=" * round(train_accu)))
    optimizer.param_groups[0]['lr'] *= lrrt_factor
    lr = optimizer.param_groups[0]['lr']
    model = torch.load(filename_base+'_model0.pt')
    criterion, optimizer = set_loss_optimizer(model, lr, momentum)

log_train_accu = np.array(log_train_accu)/100
log_val_accu = np.array(log_val_accu)/100
results = np.transpose(np.array((points, log_lr, log_train_loss, log_val_loss,
                                 log_train_accu, log_val_accu)))
np.savetxt(filename_base + '.csv', results, fmt='%4d  %.5E  %7.4f  %7.4f  %6.4f  %6.4f')
