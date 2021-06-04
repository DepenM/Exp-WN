import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import seaborn as sns
import argparse
sns.set_style('white')

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=9192)
parser.add_argument("--type", type=str, choices=["EWN", "SWN", "no-WN"], default="EWN")
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

d=784
k=10

transform = transforms.Compose(
    [   transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_length = 60000
batch_size = 128
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
test_length = 10000
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

num_fc_layers = 2
fc_neurons = [1024]
norm_type = args.type
loss_values = ['-10', '-100', '-300']
load = True
load_dir_base = './MNIST_pruning/' + str(seed) + '/'
if norm_type == 'EWN':
    WN = True
    exp_WN = True
    load_dir_base_2 = load_dir_base + 'EWN/'
elif norm_type == 'SWN':
    WN = True
    exp_WN = False
    load_dir_base_2 = load_dir_base + 'SWN/'
else:
    WN = False
    exp_WN = False
    load_dir_base_2 = load_dir_base + 'no-WN/'

act = 'relu'
class TestNet(nn.Module):
    def __init__(self, d, k, num_fc_layers, fc_neurons, act):
        super(TestNet, self).__init__()
        if WN:
            self.w = nn.ParameterList()
            self.s = nn.ParameterList()
        else:
            self.linears = nn.ModuleList()

        for i in range(num_fc_layers):
            if i==0:
                if WN:
                    temp_param = torch.randn(d, fc_neurons[0])
                    self.w.append(nn.Parameter(temp_param/torch.norm(temp_param, dim=0, keepdim=True)))
                    if exp_WN:
                        self.s.append(nn.Parameter(torch.zeros(1, fc_neurons[0])))
                    else:
                        self.s.append(nn.Parameter(torch.ones(1, fc_neurons[0])))
                    self.b = nn.Parameter(torch.zeros(1, fc_neurons[0]))
                else:
                    self.linears.append(nn.Linear(d, fc_neurons[0], bias=True))
            elif i==num_fc_layers-1:
                if WN:
                    self.lin = nn.Linear(fc_neurons[-1], k, bias=False)
                else:
                    self.linears.append(nn.Linear(fc_neurons[-1], k, bias=False))
            else:
                if WN:
                    temp_param = torch.randn(fc_neurons[i-1], fc_neurons[i])
                    self.w.append(nn.Parameter(temp_param/torch.norm(temp_param, dim=0, keepdim=True)))
                    if exp_WN:
                        self.s.append(nn.Parameter(torch.zeros(1, fc_neurons[i])))
                    else:
                        self.s.append(nn.Parameter(torch.ones(1, fc_neurons[i])))
                else:
                    self.linears.append(nn.Linear(fc_neurons[i-1], fc_neurons[i], bias=False))

        if not WN:
            self.linears.apply(self.w_init)
        self.fc_layers = num_fc_layers
        self.act = act

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        for i in range(self.fc_layers):
            if WN:
                if i!=self.fc_layers-1:
                    w1 = self.w[i]
                    if exp_WN:
                        w1 = (torch.exp(self.s[i]) / torch.norm(w1, dim=0, keepdim=True)) * w1
                    else:
                        w1 = (self.s[i] / torch.norm(w1, dim=0, keepdim=True)) * w1

                    if i==0:
                        x = torch.matmul(x, w1) + self.b
                    else:
                        x = torch.matmul(x, w1)
                    if i != self.fc_layers - 1 and self.act != 'linear':
                        if self.act == 'relu':
                            x = F.relu(x)
                        elif self.act == 'hardtanh':
                            x = F.hardtanh(x)
                else:
                    x = self.lin(x)
            else:
                x = self.linears[i](x)
                if i != self.fc_layers - 1 and self.act != 'linear':
                    if self.act == 'relu':
                        x = F.relu(x)
                    elif self.act == 'hardtanh':
                        x = F.hardtanh(x)

        return x

    def w_init(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    for ind1, curr_loss_val in enumerate(loss_values):
        if load:
            load_dir = load_dir_base_2 + str(curr_loss_val)
            net = TestNet(d, k, num_fc_layers, fc_neurons, act)
            net.to(device)
            net.load_state_dict(torch.load(load_dir + '/weights.pth'))
            last_log_loss = torch.tensor(float(curr_loss_val))

        for ind2 in range(ind1 + 2):
            if ind2==0:
                filename = load_dir + '/weight_layer_norm.txt'
            elif ind2==1:
                filename = load_dir + '/weight_layer_norm_init.txt'
            else:
                filename = load_dir + '/weight_layer_norm_' + str(loss_values[ind2-2]) + '.txt'
            test_acc = []
            cutoffs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
            for cutoff in cutoffs:
                print(cutoff)
                nums = []
                for j in range(num_fc_layers - 1):
                    nums.append(int(cutoff*fc_neurons[j]/100))
                nums.append(0)
                fin_norms = []
                with open(filename, 'r') as f:
                    for i in range(num_fc_layers):
                        norms = []
                        while True:
                            curr_norm = f.readline()
                            if curr_norm == '\n' or curr_norm == '':
                                break
                            curr_norm = curr_norm.strip('\n')
                            norms.append(float(curr_norm))
                        norms = np.array(norms)
                        fin_norms.append(norms)

                indices = []
                for i in range(num_fc_layers):
                    temp = fin_norms[i]
                    indices.append(np.array(np.sort(np.argsort(temp)[nums[i]:])))

                new_fc_neurons = []
                for i in range(num_fc_layers-1):
                    new_fc_neurons.append(len(indices[i]))

                net2 = TestNet(d, k, num_fc_layers, new_fc_neurons, act)
                net2.to(device)
                state_dict = net2.state_dict()
                curr_index_lin_s = 0
                curr_index_lin_w = 0
                for name, param in net.named_parameters():
                    if WN:
                        if 'b' in name:
                            if len(indices[0])==1:
                                temp = indices[0][0]
                                state_dict['b'].copy_(param[:, temp:temp+1])
                            else:
                                state_dict['b'].copy_(param[:, indices[0]])
                        elif 'lin' in name:
                            state_dict['lin.weight'].copy_(param[:, indices[-2]])
                        elif 's' in name:
                            if len(indices[curr_index_lin_s]) == 1:
                                temp = indices[curr_index_lin_s][0]
                                state_dict['s.' + str(curr_index_lin_s)].copy_(param[:, temp:temp + 1])
                            else:
                                state_dict['s.' + str(curr_index_lin_s)].copy_(param[:, indices[curr_index_lin_s]])
                            curr_index_lin_s += 1
                        else:
                            if len(indices[curr_index_lin_w]) == 1:
                                temp = indices[curr_index_lin_w][0]
                                if curr_index_lin_w == 0:
                                    state_dict['w.' + str(curr_index_lin_w)].copy_(param[:, temp:temp + 1])
                                else:
                                    state_dict['w.' + str(curr_index_lin_w)].copy_(param[indices[curr_index_lin_w - 1], temp:temp + 1])
                            else:
                                if curr_index_lin_w == 0:
                                    temp = param[:, indices[curr_index_lin_w]]
                                    state_dict['w.' + str(curr_index_lin_w)].copy_(temp)
                                else:
                                    temp = param[indices[curr_index_lin_w - 1]]
                                    temp2 = temp[:, indices[curr_index_lin_w]]
                                    state_dict['w.' + str(curr_index_lin_w)].copy_(temp2)
                            curr_index_lin_w += 1
                    else:
                        if 'bias' in name:
                            state_dict['linears.0.bias'].copy_(param[indices[0]])
                        elif len(indices[curr_index_lin_w]) == 1:
                            temp = indices[curr_index_lin_w][0]
                            if curr_index_lin_w == 0:
                                state_dict['linears.' + str(curr_index_lin_w) + '.weight'].copy_(param[temp:temp + 1, :])
                            else:
                                state_dict['linears.' + str(curr_index_lin_w) + '.weight'].copy_(
                                    param[temp:temp + 1, indices[curr_index_lin_w - 1]])
                            curr_index_lin_w += 1
                        else:
                            if curr_index_lin_w == 0:
                                temp = param[indices[curr_index_lin_w]]
                                state_dict['linears.' + str(curr_index_lin_w) + '.weight'].copy_(temp)
                            else:
                                temp = param[indices[curr_index_lin_w]]
                                temp2 = temp[:, indices[curr_index_lin_w - 1]]
                                state_dict['linears.' + str(curr_index_lin_w) + '.weight'].copy_(temp2)
                            curr_index_lin_w += 1
                acc = 0.0
                running_loss = 0.0
                temp_vec = torch.zeros(10, k - 1).type(torch.LongTensor).to(device)
                for j in range(10):
                    temp_index = 0
                    for l in range(k):
                        if l == j:
                            continue
                        else:
                            temp_vec[j][temp_index] = l
                            temp_index += 1

                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net2(inputs)
                    temp = outputs.gather(1, labels.unsqueeze(1))
                    outputs2 = temp - outputs
                    temp_vec_2 = temp_vec[labels]
                    s_n = outputs2.gather(1, temp_vec_2)
                    s_n_min = None
                    if torch.min(s_n) > 0:
                        s_n_min = torch.min(s_n)
                    if torch.min(s_n) > 30:
                        loss = torch.mean(torch.exp(torch.logsumexp(-s_n, dim=1, keepdim=True) - last_log_loss))
                    else:
                        loss = torch.mean(torch.log1p(torch.sum(torch.exp(-s_n), dim=1)) * torch.exp(-last_log_loss))

                    class_pred = torch.argmax(outputs, dim=1)
                    acc = acc + torch.sum(class_pred == labels)
                    running_loss = running_loss + loss.item()*inputs.shape[0]

                print('Training acc: ' + str(acc.item() / train_length))

                criterion = torch.nn.functional.cross_entropy
                net2 = net2.eval()
                acc = 0.0
                total_loss = 0.0
                with torch.no_grad():
                    for i, data in enumerate(testloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = net2(inputs)

                        loss = criterion(outputs, labels, reduction='sum')
                        total_loss += loss.item()
                        class_pred = torch.argmax(outputs, dim=1)
                        acc = acc + torch.sum(class_pred == labels)

                print('Test acc: ' + str(acc.item()/test_length))
                test_acc.append(acc.item()/test_length)

            str_test_acc = []
            for i in range(len(test_acc)):
                str_test_acc.append(str(test_acc[i]))
            if WN:
                if exp_WN:
                    mid_name = 'EWN'
                else:
                    mid_name = 'SWN'
            else:
                mid_name = 'no-WN'
            if ind2 == 0:
                end_name = '0'
            elif ind2 == 1:
                end_name = 'init'
            else:
                end_name = str(loss_values[ind2 - 2])
            name = load_dir_base + 'test_acc_prune_' + mid_name + '_diff_' + end_name + '_' + str(curr_loss_val) + '.txt'
            with open(name, 'w') as f:
                f.write('\n'.join(str_test_acc))
            f.close()

