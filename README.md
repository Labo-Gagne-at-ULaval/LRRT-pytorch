# LRRT-pytorch

Summary : Learning rate range test (LRRT) for PyTorch projects.

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
The input file should be in the current directory, and be specific for one particular test.

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
