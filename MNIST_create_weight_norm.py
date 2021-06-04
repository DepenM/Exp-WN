import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=9192)
parser.add_argument("--type", type=str, choices=["EWN", "SWN", "no-WN"], default="EWN")
args = parser.parse_args()

seed = args.seed
type = args.type
dir = './MNIST_pruning/' + str(seed) + '/' + type + '/'
losses = ['-10', '-100', '-300']
num_layers = 2

for ind1, l in enumerate(losses):
    base_name  = dir + str(l) + '/weight_layer_norm'
    for j in range(ind1 + 2):
        if j==0:
            name = base_name + '.txt'
        elif j==1:
            name = base_name + '_init.txt'
        else:
            name = base_name + '_' + str(losses[j-2]) + '.txt'
        for i in range(num_layers):
            v = np.load(dir + str(l) + '/weight_norm_' + str(i) + '.npz')
            if j==1:
                v1 = np.load(dir + str(losses[0]) + '/weight_norm_' + str(i) + '.npz')
            elif j>=2:
                v1 = np.load(dir + str(losses[j-2]) + '/weight_norm_' + str(i) + '.npz')
            str_w_norm = []
            if type=='no-WN':
                for k in range(v['w_norm'].shape[0]):
                    if j==0:
                        str_w_norm.append(str(v['w_norm'][k, -1]))
                    elif j==1:
                        str_w_norm.append(str(v['w_norm'][k, -1] - v1['w_norm'][k, 0]))
                    else:
                        str_w_norm.append(str(v['w_norm'][k, -1] - v1['w_norm'][k, -1]))
            else:
                for k in range(v['w_norm'].shape[1]):
                    if j==0:
                        str_w_norm.append(str(v['w_norm'][-1, k]))
                    elif j==1:
                        str_w_norm.append(str(v['w_norm'][-1, k] - v1['w_norm'][0, k]))
                    else:
                        str_w_norm.append(str(v['w_norm'][-1, k] - v1['w_norm'][-1, k]))

            if i==0:
                with open(name, 'w') as f:
                    f.write('\n'.join(str_w_norm))
                    f.write('\n\n')
            else:
                with open(name, 'a+') as f:
                    f.write('\n'.join(str_w_norm))
                    f.write('\n\n')
        f.close()