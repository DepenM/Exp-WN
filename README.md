# Exp-WN

This repository provides the code for experiments in the paper. The description of various files and folders is provided below:

1. lin-sep.py: Train the model and generate the logs for the lin-sep experiment in the lin-sep folder.
2. simple-traj.py: Train the model and generate the logs for the simple-traj experiment in the simple-traj folder.
3. XOR.py: Train the model and generate the logs for the XOR experiment in the XOR folder.
4. conv_rate.py: Train the model and generate the logs for the convergence rate experiment in the Conv_rate folder.
5. MNIST_train.py: Train the model on MNIST dataset and generate the logs in the MNIST_pruning folder.
6. MNIST_prune.py: Use the generated logs from MNIST_train.py to prune neurons and store the corresponding test accuracy in a file named 'pruning.txt' in the                    corresponding folder.
7. plotting folder: Contains 5 ipynb files corresponding to each experiment, that produce the desired plots by parsing the generated logs. 
