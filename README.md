# EWN and SWN

This repository provides the code for experiments on EWN (exponential weight normalization) and SWN (standard weight normalization). The description of various files and folders is provided below:

* `lin-sep.py:` Train the model and generate the logs for the lin-sep experiment in the lin-sep folder.
* `simple-traj.py:` Train the model and generate the logs for the simple-traj experiment in the simple-traj folder.
* `XOR.py:` Train the model and generate the logs for the XOR experiment in the XOR folder.
* `conv_rate.py:` Train the model and generate the logs for the convergence rate experiment in the Conv_rate folder.
* `MNIST_train.py/CIFAR_train.py:` Train the model on MNIST/CIFAR dataset and generate the logs in the MNIST_pruning/CIFAR_pruning folder. This script has two arguments - seed and type. The seed argument specified the random seed and type can take one of the 3 values - 'EWN', 'SWN' or 'no-WN'.
* `MNIST_create_weight_norm.py/CIFAR_create_weight_norm.py:` Generate intermediate weight norm files from the logs, which will be used by the pruning file for deciding which neurons to prune. It also takes in the arguments of seed and type.
* `MNIST_prune.py/CIFAR_prune.py:` Use the saved model weights and the generated intermediate weight norm files to decide neurons to prune. Store the corresponding test accuracy in the folder within MNIST_pruning/CIFAR_pruning, for a given seed. It also takes in seed and type as arguments.
* `plotting folder:` Contains 6 ipynb files corresponding to each experiment, that produce the desired plots by parsing the generated logs. 

Thus, for the main pruning experiment, three files need to be run for a given seed and type - the train file, followed by create weight norm file, and finally the prune file. The plotting file uses the final output of the prune file to generate the plots.

Note that as the weights of final models in case of MNIST/CIFAR10 dataset were very large in size, therefore the MNIST/CIFAR10 pruning folder doesn't contain the weights of the trained models. The complete folder with saved weights can be found on this link - [Pruning_Data](https://drive.google.com/file/d/1DACXs3mP3sUgTjExpciyOT-vQCkK1Azq/view?usp=sharing)
