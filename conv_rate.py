import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import seaborn as sns
import torch.nn.functional as F

sns.set_style('white')

seed = 6782
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#generating training and testing data
d=2
k=2
train_length = 20
test_length = 100
margin_w_norm = 1
class_1_points = np.array([[1/margin_w_norm, 1]])
class_2_points = np.array([[-1/margin_w_norm, 1]])
class_1_points = np.concatenate([class_1_points, np.random.uniform(1.5, 2, (int((train_length + test_length)/2) - 1, 2))], axis=0)
temp_points = np.random.uniform(-1.5, -2, (int((train_length + test_length)/2) - 1, 2))
class_2_points = np.concatenate([class_2_points, temp_points], axis=0)

training_set = np.concatenate([class_1_points[0:int(train_length/2)], class_2_points[0:int(train_length/2)]])
training_set = torch.tensor(training_set).type(torch.FloatTensor)
train_labels = np.concatenate([np.zeros((int(train_length/2), 1)), np.ones((int(train_length/2), 1))])
train_labels = torch.tensor(train_labels).type(torch.FloatTensor)
val_set = np.concatenate([class_1_points[int(train_length/2):], class_2_points[int(train_length/2):]])
val_set = torch.tensor(val_set).type(torch.FloatTensor)
val_labels = np.concatenate([np.zeros((int(test_length/2), 1)), np.ones((int(test_length/2), 1))])
val_labels = torch.tensor(val_labels).type(torch.FloatTensor)

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

#w that every network is explicitly initialized to
w_init = np.random.randn(d,1)
num_layers = [1, 2, 3, 4]
neurons = 100
is_bias = False
norm_type = 'no-WN'
if norm_type=='exp-WN':
    WN = True
    exp_WN = True
    base_dir = 'Conv_rate/Exp-WN'
elif norm_type=='standard-WN':
    WN = True
    exp_WN = False
    base_dir = 'Conv_rate/Standard-WN'
else:
    WN = False
    exp_WN = False
    base_dir = 'Conv_rate/No-WN'
act = 'linear'
lr = 0.001
batch_size = train_length
tot_epochs = 5000
norm_diff = []
angle_diff = []
init = 'all_layer_equal'
best_val_acc = 0
last_log_loss = torch.tensor(0.0)

for n_layers in num_layers:
    class TestNet(nn.Module):
        def __init__(self, n_layers, d, neurons, is_bias, act):
            super(TestNet, self).__init__()
            self.linears = nn.ModuleList()
            if WN:
                self.s = nn.ParameterList()
                self.w = nn.ParameterList()

            for i in range(n_layers):
                if i==0:
                    if n_layers==1:
                        if WN:
                            temp_param = torch.randn(d,1)
                            self.w.append(nn.Parameter((temp_param/torch.norm(temp_param)).to(device)))
                            if exp_WN:
                                self.s.append(nn.Parameter(torch.zeros(1, 1).to(device)))
                            else:
                                self.s.append(nn.Parameter(torch.ones(1, 1).to(device)))
                        else:
                            self.linears.append(nn.Linear(d, 1, bias=is_bias))
                    else:
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
                    if WN:
                        temp_param = torch.randn(neurons, 1)
                        self.w.append(nn.Parameter((temp_param/torch.norm(temp_param)).to(device)))
                        if exp_WN:
                            self.s.append(nn.Parameter(torch.zeros(1, 1).to(device)))
                        else:
                            self.s.append(nn.Parameter(torch.ones(1, 1).to(device)))
                    else:
                        self.linears.append(nn.Linear(neurons, 1, bias=is_bias))
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

        def forward(self, x, store_params=0):
            if WN:
                w_fin = None
                for i, l in enumerate(self.w):
                    w1 = self.w[i]
                    if exp_WN:
                        w1 = (torch.exp(self.s[i])/torch.norm(w1, dim=0, keepdim=True))*w1
                    else:
                        w1 = (self.s[i]/ torch.norm(w1, dim=0, keepdim=True)) * w1
                    if w_fin is None:
                        w_fin = w1
                    else:
                        w_fin = torch.matmul(w_fin, w1)

                    x = torch.matmul(x, w1)
                    if i!=self.layers-1 and self.act!='linear':
                        if self.act=='relu':
                            x = F.relu(x)
                        elif self.act=='hardtanh':
                            x = F.hardtanh(x)
            else:
                w_fin = None
                for i, l in enumerate(self.linears):
                    x = l(x)
                    if w_fin is None:
                        w_fin = l.weight.transpose(0,1)
                    else:
                        w_fin = torch.matmul(w_fin, l.weight.transpose(0,1))

                    if i!=self.layers-1 and self.act!='linear':
                        if self.act=='relu':
                            x = F.relu(x)
                        elif self.act=='hardtanh':
                            x = F.hardtanh(x)

            if store_params:
                max_margin = margin_w_norm*np.array([[-1], [0]])
                w_fin = w_fin.detach().cpu().numpy()
                norm_diff.append(np.linalg.norm(w_fin/np.linalg.norm(w_fin) - max_margin/margin_w_norm))
                angle_diff.append(np.dot(w_fin.transpose(), max_margin)[0,0]/(np.linalg.norm(w_fin)*margin_w_norm))
            return x

    if __name__ == '__main__':
        dir = base_dir + '/layer_' + str(n_layers)
        np.set_printoptions(suppress=True)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                                                 num_workers=0)

        #Initialize WN network to exactly the same parameters as a non-WN network
        if WN:
            WN = False
            net = TestNet(n_layers, d, neurons, is_bias, act)
            net.to(device)
            for i, data in enumerate(trainloader, 0):
                inputs,labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
            copy_params = []
            for name, param in net.named_parameters():
                copy_params.append(param)
            WN = True
            net = TestNet(n_layers, d, neurons, is_bias, act)
            net.to(device)
            state_dict = net.state_dict()
            for name, param in net.named_parameters():
                if 's' in name:
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
            net = TestNet(n_layers, d, neurons, is_bias, act)
            net.to(device)

        #Initialize all multilayer nets in function space to the same point
        if init=='all_layer_equal':
            if WN:
                s_temp = []
                w_temp = []
                for name, param in net.named_parameters():
                    if 's' in name:
                        s_temp.append(param)
                    else:
                        w_temp.append(param)
                w_overall=None
                for i in range(len(s_temp)-1):
                    if exp_WN:
                        if w_overall is None:
                            w_overall = (torch.exp(s_temp[i]) / torch.norm(w_temp[i], dim=0, keepdim=True)) * w_temp[i]
                        else:
                            w_overall = torch.matmul(w_overall, (torch.exp(s_temp[i]) / torch.norm(w_temp[i], dim=0, keepdim=True)) * w_temp[i])
                    else:
                        if w_overall is None:
                            w_overall = (s_temp[i] / torch.norm(w_temp[i], dim=0, keepdim=True)) * w_temp[i]
                        else:
                            w_overall = torch.matmul(w_overall, (s_temp[i] / torch.norm(w_temp[i], dim=0, keepdim=True)) * w_temp[i])

                if w_overall is None:
                    fin_layer = w_init
                else:
                    w_overall = w_overall.detach().cpu().numpy()
                    fin_layer = np.matmul(np.linalg.pinv(w_overall), w_init)
                state_dict = net.state_dict()
                if exp_WN:
                    state_dict['s.' + str(n_layers-1)].copy_(torch.tensor([[np.log(np.linalg.norm(fin_layer))]]))
                else:
                    state_dict['s.' + str(n_layers - 1)].copy_(torch.tensor([[np.linalg.norm(fin_layer)]]))
                state_dict['w.' + str(n_layers-1)].copy_(torch.tensor(np.reshape(fin_layer, [-1,1])))
            else:
                w_overall = None
                for name, param in net.named_parameters():
                    ind = int(name.split('.')[1])
                    if ind < n_layers-1:
                        if w_overall is None:
                            w_overall = param.transpose(0,1)
                        else:
                            w_overall = torch.matmul(w_overall, param.transpose(0,1))

                if w_overall is None:
                    fin_layer = w_init
                else:
                    w_overall = w_overall.detach().cpu().numpy()
                    fin_layer = np.matmul(np.linalg.pinv(w_overall), w_init)
                state_dict = net.state_dict()
                state_dict['linears.' + str(n_layers-1) + '.weight'].copy_(torch.tensor(np.reshape(fin_layer, [1, -1])))

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
                outputs = net(inputs, 1)
                loss = criterion(outputs, labels, reduction='sum')
                class_pred = outputs > 0
                acc = acc + torch.sum(class_pred == labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss.append(running_loss / (i + 1))
            train_acc.append(acc.item()/train_length)

            # print('\n\n')
            print('[%d] Train loss: %.3f, Train acc: %.3f' %
                  (epoch + 1, running_loss / (i + 1), acc.item()/train_length))

            # print('\n\n')
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
                  (epoch + 1, total_loss / (i + 1), acc.item()/test_length))

            test_loss.append(total_loss / (i + 1))
            test_acc.append(acc.item()/test_length)

            if acc.item()/test_length > best_val_acc:
                best_val_acc = acc.item()/test_length

    torch.save(net.state_dict(), dir + '/weights.pth')

    print('Best_val_acc is ' + str(best_val_acc))

    plt.plot(list(range(len(train_loss))), train_loss, label='train_loss')
    plt.plot(list(range(len(train_loss))), test_loss, label='test_loss')
    plt.legend(loc='upper right')
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

    plt.plot(list(range(len(train_acc))), train_acc, label='train_acc')
    plt.plot(list(range(len(test_acc))), test_acc, label='test_acc')
    plt.legend(loc='upper right')
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

    str_diff_norm = []
    for i in range(len(norm_diff)):
        str_diff_norm.append(str(norm_diff[i]))
    name = dir + '/diff_norm.txt'
    with open(name, 'w') as f:
        f.write('\n'.join(str_diff_norm))
    f.close()

    str_angle_diff = []
    for i in range(len(angle_diff)):
        str_angle_diff.append(str(angle_diff[i]))
    name = dir + '/angle_diff.txt'
    with open(name, 'w') as f:
        f.write('\n'.join(str_angle_diff))
    f.close()