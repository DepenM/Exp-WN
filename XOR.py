import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import seaborn as sns
sns.set_style('white')

np.random.seed(9156)
torch.manual_seed(9156)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#generating training and testing data
d=2
k=2
train_length = 4
test_length = 4
class_1_points = np.array([[1,1], [-1,-1], [1,1], [-1,-1]])
class_2_points = np.array([[1,-1], [-1,1], [1,-1], [-1,1]])

training_set = np.concatenate([class_1_points[0:int(train_length/2)], class_2_points[0:int(train_length/2)]])
training_set = torch.tensor(training_set).type(torch.FloatTensor)
train_labels = np.concatenate([np.zeros((int(train_length/2), 1)), np.ones((int(train_length/2), 1))])
train_labels = torch.tensor(train_labels).type(torch.FloatTensor)
val_set = np.concatenate([class_1_points[int(train_length/2):], class_2_points[int(train_length/2):]])
val_set = torch.tensor(val_set).type(torch.FloatTensor)
val_labels = np.concatenate([np.zeros((int(test_length/2), 1)), np.ones((int(test_length/2), 1))])
val_labels = torch.tensor(val_labels).type(torch.FloatTensor)

plt.scatter(class_1_points[0:int(train_length / 2), 0], class_1_points[0:int(train_length / 2), 1], c='r')
plt.scatter(class_2_points[0:int(train_length / 2), 0], class_2_points[0:int(train_length / 2), 1], c='b')
plt.show()


class MyCustomDataset(Dataset):
    """ Linearly separable dataset."""

    def __init__(self, train_data, test_data, train=True, transform=None):
        """
        Args:
            train_data: training data
            test_data: testing data
            train: whether training data required or nor
        """
        self.train_data = train_data
        self.test_data = test_data
        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return (self.train_data[0].shape[0])
        else:
            return (self.test_data[0].shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            sample = (self.train_data[0][idx, :], self.train_data[1][idx])
        else:
            sample = (self.test_data[0][idx, :], self.test_data[1][idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

dataset_train = MyCustomDataset([training_set,train_labels],[],True)
dataset_test = MyCustomDataset([],[val_set, val_labels],False)

num_layers = 2
neurons = 20
act = 'relu'
norm_type = 'no-WN'
"""
frac governs the learning rate order
frac = 5 means lr = 1/(L^(1 - 1/5))
frac = 0 means lr = 1/L
"""
if norm_type=='exp-WN':
    WN = True
    exp_WN = True
    dir = 'XOR/Exp-WN'
    lr_limit = 0.1
    frac = 5
elif norm_type=='standard-WN':
    WN = True
    exp_WN = False
    dir = 'XOR/Standard-WN'
    lr_limit = 1
    frac = 15
else:
    WN = False
    exp_WN = False
    dir = 'XOR/No-WN'
    lr_limit = 0.1
    frac = 15
is_bias = False
lr = 0.01
r_u = 1.1
r_d = 1.1
batch_size = train_length
tot_epochs = 100000
log_loss_limit = 50
total_weights = (num_layers-1)*neurons
weight_norms = []
for i in range(total_weights):
    weight_norms.append([])

last_log_loss = torch.tensor(0.0)

class TestNet(nn.Module):
    def __init__(self, n_layers, d, neurons, is_bias, act):
        super(TestNet, self).__init__()
        self.linears = nn.ModuleList()
        if WN:
            self.s = nn.ParameterList()
            self.w = nn.ParameterList()

        for i in range(n_layers):
            if i==0:
                if WN:
                    temp_param = torch.randn(d, neurons)
                    temp_param = (1/torch.norm(temp_param, dim=0, keepdim=True))*temp_param
                    self.w.append(nn.Parameter(temp_param.to(device)))
                    if exp_WN:
                        self.s.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
                    else:
                        self.s.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                else:
                    self.linears.append(nn.Linear(d, neurons, bias=is_bias))
            elif i==n_layers-1:
                temp_w = np.ones((int(neurons/2), 1))
                temp_w = np.concatenate([temp_w, np.ones((int(neurons/2), 1))*-1], axis=0)
                self.fixed_w = torch.tensor(temp_w).type(torch.FloatTensor).to(device)
            else:
                if WN:
                    temp_param = torch.randn(neurons, neurons)
                    temp_param = (1 / torch.norm(temp_param, dim=0, keepdim=True)) * temp_param
                    self.w.append(nn.Parameter(temp_param.to(device)))
                    if exp_WN:
                        self.s.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
                    else:
                        self.s.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                else:
                    self.linears.append(nn.Linear(neurons, neurons, bias=is_bias))
        self.layers = n_layers
        self.act = act

    def forward(self, x):
        if WN:
            for i, l in enumerate(self.w):
                w1 = self.w[i]
                if exp_WN:
                    w1 = (torch.exp(self.s[i])/torch.norm(w1, dim=0, keepdim=True))*w1
                else:
                    w1 = (self.s[i]/ torch.norm(w1, dim=0, keepdim=True)) * w1

                x = torch.matmul(x, w1)

                if i!=self.layers-1 and self.act!='linear':
                    if self.act=='relu':
                        x = F.relu(x)
                    elif self.act=='hardtanh':
                        x = F.hardtanh(x)

            x = torch.matmul(x, self.fixed_w)
        else:
            for i, l in enumerate(self.linears):
                x = l(x)
                if i!=self.layers-1 and self.act!='linear':
                    if self.act=='relu':
                        x = F.relu(x)
                    elif self.act=='hardtanh':
                        x = F.hardtanh(x)
            x = torch.matmul(x, self.fixed_w)

        return x

    def w_init(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal(m.weight)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                             num_workers=0)

    if WN:
        WN = False
        net = TestNet(num_layers, d, neurons, is_bias, act)
        net.to(device)
        for i, data in enumerate(trainloader, 0):
            inputs,labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
        copy_params = []
        for name, param in net.named_parameters():
            copy_params.append(param)
        WN = True
        net = TestNet(num_layers, d, neurons, is_bias, act)
        net.to(device)
        state_dict = net.state_dict()
        for name, param in net.named_parameters():
            if 'linear' in name:
                state_dict[name].copy_(copy_params[-1])
            elif 's' in name:
                ind = int(name.split('.')[1])
                if exp_WN:
                    state_dict[name].copy_(torch.log(torch.norm(copy_params[ind], dim=1, keepdim=True).transpose(0,1)))
                else:
                    state_dict[name].copy_(
                        torch.norm(copy_params[ind], dim=1, keepdim=True).transpose(0, 1))
            else:
                ind = int(name.split('.')[1])
                state_dict[name].copy_((copy_params[ind]/torch.norm(copy_params[ind], dim=1, keepdim=True)).transpose(0,1))

        outputs2 = net(inputs)
        #print(torch.norm(outputs-outputs2))
    else:
        net = TestNet(num_layers, d, neurons, is_bias, act)
        net.to(device)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = optim.SGD(net.parameters(), lr)
    best_val_acc = 0
    for epoch in range(tot_epochs):  # loop over the dataset multiple times
        net = net.train()
        running_loss = 0.0
        acc = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            labels_new = 2*labels - 1
            labels_new = labels_new.type(torch.FloatTensor).to(device)
            s_n = labels_new*outputs
            s_n_min = None
            if torch.min(s_n) > 30:
                loss = torch.sum(torch.exp(-s_n-last_log_loss))
            else:
                loss = torch.sum(torch.log1p(torch.exp(-s_n)))*torch.exp(-last_log_loss)

            class_pred = outputs >= 0
            acc = acc + torch.sum(class_pred == labels)

            loss.backward()
            if num_layers>=2:
                if WN:
                    w_s = []
                    s_s = []
                    fin_ws = []
                    for name, param in net.named_parameters():
                        if 's' in name:
                            s_s.append(param)
                        else:
                            w_s.append(param)

                    for j in range(num_layers-1):
                        if exp_WN:
                            fin_ws.append(
                                (torch.exp(s_s[j]) / torch.norm(w_s[j], dim=0, keepdim=True)) * w_s[j])
                        else:
                            fin_ws.append((s_s[j] / torch.norm(w_s[j], dim=0, keepdim=True)) * w_s[j])
                else:
                    fin_ws = []
                    for name, param in net.named_parameters():
                        ind = int(name.split('.')[1])
                        fin_ws.append(param.transpose(0,1))

                for j in range(total_weights):
                    ind1 = int(j / neurons)
                    ind2 = j % neurons
                    weight_norms[j].append(torch.norm(fin_ws[ind1][:, ind2]).detach().cpu().numpy())

            while(True):
                for g in optimizer.param_groups:
                    if frac==0:
                        g['lr'] = lr
                    else:
                        g['lr'] = lr*torch.exp(last_log_loss/frac)

                optimizer.step()

                for j, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)
                labels_new_3 = 2 * labels - 1
                labels_new_3 = labels_new_3.type(torch.FloatTensor).to(device)
                s_n_3 = labels_new_3 * outputs
                new_last_log_loss = torch.log(loss.detach()) + last_log_loss
                if torch.min(s_n_3) > 30:
                    loss_new = torch.sum(torch.exp(-s_n_3 - last_log_loss))
                    loss_new_2 = torch.sum(torch.exp(-s_n_3 - new_last_log_loss))
                else:
                    loss_new = torch.sum(torch.log1p(torch.exp(-s_n_3))) * torch.exp(-last_log_loss)
                    loss_new_2 = torch.sum(torch.log1p(torch.exp(-s_n_3))) * torch.exp(-new_last_log_loss)

                if loss_new>loss or loss_new_2==0:
                    for g in optimizer.param_groups:
                        if frac==0:
                            g['lr'] = -lr
                        else:
                            g['lr'] = -lr*torch.exp(last_log_loss/frac)
                    optimizer.step()
                    lr = lr/r_d
                    if lr == 0:
                        break
                else:
                    if lr<lr_limit:
                        lr = lr*r_u
                    break

            last_log_loss = torch.log(loss.detach()) + last_log_loss
            running_loss += loss.item()

        train_loss.append(last_log_loss.detach().cpu().numpy())
        train_acc.append(acc.item()/train_length)

        print('[%d] Train loss: %.3f, Train acc: %.3f' %
              (epoch + 1, last_log_loss.detach().cpu().numpy(), acc.item()/train_length))

        net = net.eval()
        acc = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels, reduction='sum')
                total_loss += loss.item()
                class_pred = outputs > 0
                acc = acc + torch.sum(class_pred == labels)
        print('[%d] Validation loss: %.3f, Validation acc: %.3f' %
              (epoch + 1, total_loss/test_length, acc.item()/test_length))

        test_loss.append(total_loss / test_length)
        test_acc.append(acc.item()/test_length)

        if last_log_loss < -log_loss_limit or lr==0:
            break

        if acc.item()/test_length > best_val_acc:
            best_val_acc = acc.item()/test_length

torch.save(net.state_dict(), dir + '/weights.pth')

name = dir + '/fin_log_loss.txt'
with open(name, 'w') as f:
    f.write(str(last_log_loss))
f.close()

print('Best_val_acc for lr = ' + str(lr) + ' is ' + str(best_val_acc))

font_size = 14
line_width = 3

plt.plot(list(range(len(train_loss))), train_loss, linewidth=line_width)
plt.xlabel('Steps', fontsize=font_size)
plt.ylabel('Train loss(in log)', fontsize=font_size)
plt.tight_layout()
plt.show()

plt.plot(list(range(len(test_loss))), test_loss, linewidth=line_width)
plt.xlabel('Steps', fontsize=font_size)
plt.ylabel('Test loss', fontsize=font_size)
plt.tight_layout()
plt.show()

str_train_loss = []
for i in range(len(train_loss)):
    str_train_loss.append(str(train_loss[i]))
name = dir + '/train_loss.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_train_loss))
f.close()

str_test_loss = []
for i in range(len(test_loss)):
    str_test_loss.append(str(test_loss[i]))
name = dir + '/test_loss.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_test_loss))
f.close()

plt.plot(list(range(len(train_acc))), train_acc, label='train_acc', linewidth=line_width)
plt.plot(list(range(len(test_acc))), test_acc, label='test_acc', linewidth=line_width)
plt.legend(loc='upper right')
plt.xlabel('Steps', fontsize=font_size)
plt.ylabel('Accuracy', fontsize=font_size)
plt.tight_layout()
plt.show()

str_train_acc = []
for i in range(len(train_acc)):
    str_train_acc.append(str(train_acc[i]))
name = dir + '/train_acc.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_train_acc))
f.close()

str_test_acc = []
for i in range(len(test_acc)):
    str_test_acc.append(str(test_acc[i]))
name = dir + '/test_acc.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_test_acc))
f.close()


for j in range(total_weights):
    name = dir + '/w_norm_' + str(j) + '.npz'
    with open(name, 'wb') as f:
        np.savez(f, w_norm=np.array(weight_norms[j]))

