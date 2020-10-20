import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import seaborn as sns
sns.set_style('white')

seed = 2137
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#generating training and testing data
d=2
k=2
train_length = 4
test_length = 100
margin_w_norm = 1
class_1_points = np.array([[1/margin_w_norm, 1]])
class_2_points = np.array([[-1/margin_w_norm, 1]])
min_val_non_supp = np.random.uniform(1/margin_w_norm, 1.5/margin_w_norm)
max_val_non_supp = np.random.uniform(1.5/margin_w_norm, 2/margin_w_norm)
class_1_points = np.concatenate([class_1_points, np.random.uniform(min_val_non_supp, max_val_non_supp, (int((train_length + test_length)/2) - 1, 2))], axis=0)
temp_points = np.random.uniform(-min_val_non_supp, -max_val_non_supp, (int((train_length + test_length)/2) - 1, 2))
class_2_points = np.concatenate([class_2_points, temp_points], axis=0)

training_set = np.concatenate([class_1_points[0:int(train_length/2)], class_2_points[0:int(train_length/2)]])
training_set = torch.tensor(training_set).type(torch.FloatTensor)
train_labels = np.concatenate([np.zeros((int(train_length/2), 1)), np.ones((int(train_length/2), 1))])
train_labels = torch.tensor(train_labels).type(torch.FloatTensor)
val_set = np.concatenate([class_1_points[int(train_length/2):], class_2_points[int(train_length/2):]])
val_set = torch.tensor(val_set).type(torch.FloatTensor)
val_labels = np.concatenate([np.zeros((int(test_length/2), 1)), np.ones((int(test_length/2), 1))])
val_labels = torch.tensor(val_labels).type(torch.FloatTensor)

plt.scatter(class_1_points[0:int(train_length/2)][:,0], class_1_points[0:int(train_length/2)][:,1], c='red', s=40, label='class -1')
plt.scatter(class_2_points[0:int(train_length/2)][:,0], class_2_points[0:int(train_length/2)][:,1], c='blue', s=40, label='class 1')
plt.legend(prop={'size':14})
plt.locator_params(axis='y', nbins=4)
plt.locator_params(axis='x', nbins=4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Dataset')
plt.tight_layout()
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
neurons = 8
WN = True
exp_WN = False
"""
frac governs the learning rate order
frac = 30 means lr = 1/(L^(1 - 1/30))
frac = 0 means lr = 1/L
"""
frac = 30
is_bias = False
act = 'relusq'
lr = 0.01
lr_limit = 0.01
r_u = 1.1
r_d = 1.1
batch_size = train_length
tot_epochs = 40000
#Final log loss value
log_loss_limit = 300
param_margin = []
w_dir = []
fin_l_tilde = None
loss_vec_dir = None
full_grad_dir = []

if WN:
    if exp_WN:
        dir = 'lin-sep/Exp-WN/'
    else:
        dir = 'lin-sep/Standard-WN/'

total_weights = (num_layers-1)*neurons + 1
if num_layers >= 2:
    random_weight_pairs = int((total_weights*(total_weights-1))/2)
    rand_indices = []
    for i in range(total_weights):
        for j in range(i+1,total_weights):
            rand_indices.append(i)
            rand_indices.append(j)
    rand_indices = np.array(rand_indices)
    rand_grad_norms = []
    rand_grad_lr_norms = []
    rand_weight_norms = []
    rand_w_grad_angle = []
    for i in range(2*random_weight_pairs):
        rand_grad_norms.append([])
        rand_grad_lr_norms.append([])
        rand_weight_norms.append([])
        rand_w_grad_angle.append([])

ratio = []
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

    def forward(self, x):
        if WN:
            for i in range(self.layers):
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
                    elif self.act=='relusq':
                        x = (F.relu(x))**2
        else:
            for i, l in enumerate(self.linears):
                x = l(x)
                if i!=self.layers-1 and self.act!='linear':
                    if self.act=='relu':
                        x = F.relu(x)
                    elif self.act=='hardtanh':
                        x = F.hardtanh(x)
                    elif self.act=='relusq':
                        x = (F.relu(x))**2
        return x

    def w_init(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

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
    curr_index = 0
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
            if torch.min(s_n) > 0:
                s_n_min = torch.min(s_n)
                point_wise_min = torch.min(s_n, dim=1, keepdim=True).values
                point_wise_min = point_wise_min - s_n_min
                point_wise_min = torch.exp(-point_wise_min)
                point_wise_min = (point_wise_min/torch.norm(point_wise_min)).detach().cpu().numpy()
                if loss_vec_dir is None:
                    loss_vec_dir = point_wise_min
                else:
                    loss_vec_dir = np.concatenate([loss_vec_dir, point_wise_min], axis=1)
            if torch.min(s_n) > 30:
                loss = torch.sum(torch.exp(-s_n-last_log_loss))
            else:
                loss = torch.sum(torch.log1p(torch.exp(-s_n)))*torch.exp(-last_log_loss)

            class_pred = outputs > 0
            acc = acc + torch.sum(class_pred == labels)

            loss.backward()
            if num_layers>=2:
                if WN:
                    WN = False
                    net2 = TestNet(num_layers, d, neurons, is_bias, act)
                    net2.to(device)
                    curr_w_norm = None
                    w_s = []
                    s_s = []
                    fin_ws = []
                    fin_grads = []
                    for name, param in net.named_parameters():
                        if 'linear' in name:
                            temp_lin = param.transpose(0,1)
                        elif 's' in name:
                            s_s.append(param)
                        else:
                            w_s.append(param)

                    if exp_WN:
                        curr_temp_ind = 0
                        for j in range(len(w_s)):
                            for k in range(w_s[j].shape[1]):
                                if len(w_dir) <= curr_temp_ind:
                                    w_dir.append((w_s[j][:,k:k+1]/torch.norm(w_s[j][:,k:k+1])).detach().cpu().numpy())
                                else:
                                    w_dir[curr_temp_ind] = np.concatenate([w_dir[curr_temp_ind], (w_s[j][:,k:k+1]/torch.norm(w_s[j][:,k:k+1])).detach().cpu().numpy()], axis=1)
                                curr_temp_ind += 1
                    else:
                        curr_temp_ind = 0
                        for j in range(len(w_s)):
                            for k in range(w_s[j].shape[1]):
                                if len(w_dir) <= curr_temp_ind:
                                    w_dir.append((torch.sign(s_s[j][0,k])*(w_s[j][:,k:k+1]/torch.norm(w_s[j][:,k:k+1]))).detach().cpu().numpy())
                                else:
                                    w_dir[curr_temp_ind] = np.concatenate(
                                        [w_dir[curr_temp_ind], (torch.sign(s_s[j][0,k])*(w_s[j][:,k:k+1]/torch.norm(w_s[j][:,k:k+1]))).detach().cpu().numpy()], axis=1)
                                curr_temp_ind += 1

                    for j in range(num_layers):
                        if exp_WN:
                            if curr_w_norm is None:
                                curr_w_norm = torch.norm(torch.exp(s_s[j]))**2
                            else:
                                curr_w_norm = curr_w_norm + torch.norm(torch.exp(s_s[j]))**2

                            fin_ws.append(
                                (torch.exp(s_s[j]) / torch.norm(w_s[j], dim=0, keepdim=True)) * w_s[j])
                        else:
                            if curr_w_norm is None:
                                curr_w_norm = torch.norm(s_s[j])**2
                            else:
                                curr_w_norm = curr_w_norm + torch.norm(s_s[j])**2

                            fin_ws.append((s_s[j] / torch.norm(w_s[j], dim=0, keepdim=True)) * w_s[j])

                    WN = True
                    for j, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = net(inputs)
                    state_dict = net2.state_dict()
                    for name, param in net2.named_parameters():
                        ind = int(name.split('.')[1])
                        state_dict[name].copy_(fin_ws[ind].transpose(0, 1))

                    WN = False
                    labels_new_2 = 2*labels - 1
                    outputs2 = net2(inputs)
                    #print(torch.norm(outputs - outputs2))
                    s_n_2 = labels_new_2 * outputs2
                    if torch.min(s_n_2) > 30:
                        loss2 = torch.sum(torch.exp(-s_n_2 - last_log_loss))
                    else:
                        loss2 = torch.sum(torch.log1p(torch.exp(-s_n_2))) * torch.exp(-last_log_loss)

                    loss2.backward()
                    fin_grad_vec = None
                    for name, param in net2.named_parameters():
                        fin_grads.append(param.grad.transpose(0,1))
                        if fin_grad_vec is None:
                            fin_grad_vec = param.grad.reshape(-1,1).detach().cpu().numpy()
                        else:
                            fin_grad_vec = np.concatenate([fin_grad_vec, param.grad.reshape(-1,1).detach().cpu().numpy()], axis=0)

                    fin_grad_vec = fin_grad_vec/np.linalg.norm(fin_grad_vec)
                    WN = True
                else:
                    curr_w_norm = None
                    fin_ws = []
                    fin_grads = []
                    fin_grad_vec = None
                    for name, param in net.named_parameters():
                        ind = int(name.split('.')[1])
                        if curr_w_norm is None:
                            curr_w_norm = torch.norm(param)**2
                        else:
                            curr_w_norm = curr_w_norm + torch.norm(param)**2
                        fin_ws.append(param.transpose(0,1))
                        fin_grads.append(param.grad.transpose(0,1))
                        if fin_grad_vec is None:
                            fin_grad_vec = param.grad.reshape(-1,1).detach().cpu().numpy()
                        else:
                            fin_grad_vec = np.concatenate([fin_grad_vec, param.grad.reshape(-1,1).detach().cpu().numpy()], axis=0)

                    fin_grad_vec = fin_grad_vec / np.linalg.norm(fin_grad_vec)

                    curr_temp_ind = 0
                    for j in range(len(fin_ws)):
                        for k in range(fin_ws[j].shape[1]):
                            if len(w_dir) <= curr_temp_ind:
                                w_dir.append((fin_ws[j][:, k:k+1]/torch.norm(fin_ws[j][:, k])).detach().cpu().numpy())
                            else:
                                w_dir[curr_temp_ind] = np.concatenate(
                                    [w_dir[curr_temp_ind], (fin_ws[j][:, k:k+1]/torch.norm(fin_ws[j][:, k])).detach().cpu().numpy()], axis=1)
                            curr_temp_ind += 1

                for j in range(2 * random_weight_pairs):
                    ind1 = int(rand_indices[j] / neurons)
                    ind2 = rand_indices[j] % neurons
                    rand_weight_norms[j].append(torch.norm(fin_ws[ind1][:, ind2]).detach().cpu().numpy())
                    rand_grad_norms[j].append(
                        torch.norm(fin_grads[ind1][:, ind2]).detach().cpu().numpy())
                    if torch.norm(fin_grads[ind1][:, ind2]).detach().cpu().numpy() == 0:
                        rand_w_grad_angle[j].append(0.0)
                    else:
                        temp1 = torch.dot(fin_ws[ind1][:, ind2], fin_grads[ind1][:, ind2]).detach().cpu().numpy()
                        rand_w_grad_angle[j].append(
                            temp1 / (torch.norm(fin_ws[ind1][:, ind2]) * torch.norm(fin_grads[ind1][:, ind2])).detach().cpu().numpy())
                    if frac==0:
                        rand_grad_lr_norms[j].append(
                            (torch.norm(fin_grads[ind1][:, ind2]) * lr *torch.norm(fin_ws[ind1][:, ind2])).detach().cpu().numpy())
                    else:
                        rand_grad_lr_norms[j].append((torch.norm(fin_grads[ind1][:, ind2]) * torch.exp(
                            last_log_loss / frac) * lr * torch.norm(fin_ws[ind1][:, ind2])).detach().cpu().numpy())

                if s_n_min is not None:
                    curr_w_norm = torch.sqrt(curr_w_norm)
                    if act=='relusq':
                        param_margin.append((s_n_min/torch.pow(curr_w_norm, np.power(2, num_layers)-1)).detach().cpu().numpy())
                    else:
                        param_margin.append(
                            (s_n_min / torch.pow(curr_w_norm, num_layers)).detach().cpu().numpy())

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

            if s_n_min is not None:
                if WN:
                    for j, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        labels_new = 2 * labels - 1
                        labels_new = labels_new.type(torch.FloatTensor).to(device)
                        s_n = labels_new * outputs
                        s_n_min = None
                        if torch.min(s_n) > 0:
                            s_n_min = torch.min(s_n)
                            point_wise_min = torch.min(s_n, dim=1, keepdim=True).values
                            point_wise_min = point_wise_min - s_n_min
                            point_wise_min = torch.exp(-point_wise_min)
                            point_wise_min = point_wise_min / torch.norm(point_wise_min)

                    if s_n_min is None:
                        continue
                    WN = False
                    net2 = TestNet(num_layers, d, neurons, is_bias, act)
                    net2.to(device)
                    w_s = []
                    s_s = []
                    fin_ws = []
                    fin_grads = []
                    for name, param in net.named_parameters():
                        if 'linear' in name:
                            temp_lin = param.transpose(0, 1)
                        elif 's' in name:
                            s_s.append(param)
                        else:
                            w_s.append(param)

                    total_norm = torch.tensor(0.0)
                    for j in range(num_layers):
                        if exp_WN:
                            total_norm += (torch.norm(torch.exp(s_s[j])))**2
                            fin_ws.append(
                                (torch.exp(s_s[j]) / torch.norm(w_s[j], dim=0, keepdim=True)) * w_s[j])
                        else:
                            total_norm += (torch.norm(s_s[j]))**2
                            fin_ws.append((s_s[j] / torch.norm(w_s[j], dim=0, keepdim=True)) * w_s[j])

                    total_norm = torch.sqrt(total_norm)
                    WN = True
                    for j, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = net(inputs)
                    state_dict = net2.state_dict()
                    for name, param in net2.named_parameters():
                        ind = int(name.split('.')[1])
                        state_dict[name].copy_(fin_ws[ind].transpose(0, 1))

                    WN = False
                    labels_new_2 = 2 * labels - 1
                    outputs2 = net2(inputs)
                    #print(torch.norm(outputs - outputs2))
                    optimizer2 = optim.SGD(net2.parameters(), lr)
                    optimizer2.zero_grad()
                    temp_grad_vec = None
                    new_last_log_loss = torch.log(loss.detach()) + last_log_loss
                    for j, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = net2(inputs)
                        labels_new = 2 * labels - 1
                        labels_new = labels_new.type(torch.FloatTensor).to(device)
                        s_n = labels_new * outputs
                        if torch.min(s_n) > 30:
                            loss_new = torch.sum(torch.exp(-s_n - new_last_log_loss))
                        else:
                            loss_new = torch.sum(torch.log1p(torch.exp(-s_n))) * torch.exp(-new_last_log_loss)
                    loss_new.backward()
                    for name, param in net2.named_parameters():
                        if temp_grad_vec is None:
                            temp_grad_vec = np.reshape(param.grad.detach().cpu().numpy(), -1)
                        else:
                            temp_grad_vec = np.concatenate([temp_grad_vec, np.reshape(param.grad.detach().cpu().numpy(), -1)])

                    full_grad_dir.append(temp_grad_vec)
                    optimizer2.zero_grad()
                    temp_l_tilde = None
                    for j, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        for k in range(train_length):
                            total_norm_per_data = torch.tensor(0.0)
                            outputs = net2(inputs[k:k+1])
                            outputs[0,0].backward()
                            curr_index = 0
                            temp_l_tilde_1 = None
                            for name, param in net2.named_parameters():
                                if temp_l_tilde_1 is None:
                                    temp_l_tilde_1 = np.reshape((labels_new_2[k]*point_wise_min[k]*param.grad).detach().cpu().numpy(), -1)
                                else:
                                    temp_l_tilde_1 = np.concatenate([temp_l_tilde_1, np.reshape((labels_new_2[k]*point_wise_min[k]*param.grad).detach().cpu().numpy(), -1)])

                                total_norm_per_data += torch.norm(param.grad)**2
                            if temp_l_tilde is None:
                                temp_l_tilde = temp_l_tilde_1.copy()
                            else:
                                temp_l_tilde += temp_l_tilde_1

                            optimizer2.zero_grad()

                    fin_l_tilde = temp_l_tilde.copy()
                    WN = True

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

name = dir + 'fin_log_loss.txt'
with open(name, 'w') as f:
    f.write(str(last_log_loss))
f.close()

print('Best_val_acc is ' + str(best_val_acc))

font_size = 14
line_width = 3

plt.plot(list(range(len(train_loss))), train_loss, linewidth = line_width)
plt.xlabel('Steps', fontsize=font_size)
plt.ylabel('Train loss(in log)', fontsize=font_size)
plt.title('Train Loss')
plt.tight_layout()
plt.show()

plt.plot(list(range(len(test_loss))), test_loss, linewidth = line_width)
plt.xlabel('Steps', fontsize=font_size)
plt.ylabel('Test loss', fontsize=font_size)
plt.title('Test Loss')
plt.tight_layout()
plt.show()

str_train_loss = []
for i in range(len(train_loss)):
    str_train_loss.append(str(train_loss[i]))
name = dir + 'train_loss.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_train_loss))
f.close()

str_test_loss = []
for i in range(len(test_loss)):
    str_test_loss.append(str(test_loss[i]))
name = dir + 'test_loss.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_test_loss))
f.close()

plt.plot(list(range(len(train_acc))), train_acc, label='train_acc', linewidth = line_width)
plt.plot(list(range(len(test_acc))), test_acc, label='test_acc', linewidth = line_width)
plt.legend(loc='upper right')
plt.xlabel('Steps', fontsize=font_size)
plt.ylabel('Accuracy', fontsize=font_size)
plt.title('Accuracy')
plt.tight_layout()
plt.show()

str_train_acc = []
for i in range(len(train_acc)):
    str_train_acc.append(str(train_acc[i]))
name = dir + 'train_acc.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_train_acc))
f.close()

str_test_acc = []
for i in range(len(test_acc)):
    str_test_acc.append(str(test_acc[i]))
name = dir + 'test_acc.txt'
with open(name, 'w') as f:
    f.write('\n'.join(str_test_acc))
f.close()

for j in range(len(w_dir)):
    name = dir + 'w_dir_' + str(j) + '.npz'
    with open(name, 'wb') as f:
        np.savez(f, w_dir=w_dir[j])

name = dir + 'param_margin.npz'
with open(name, 'wb') as f:
    np.savez(f, param_margin=np.array(param_margin))

for i in range(total_weights):
    if i==0:
        name = dir + 'w_norm_' + str(i) + '.npz'
        with open(name, 'wb') as f:
            np.savez(f, w_norm=rand_weight_norms[i], grad_norm=rand_grad_norms[i],
                     grad_lr_norm=rand_grad_lr_norms[i], w_grad_angle=rand_w_grad_angle[i])
    else:
        name = dir + 'w_norm_' + str(i) + '.npz'
        with open(name, 'wb') as f:
            np.savez(f, w_norm=rand_weight_norms[2*i - 1], grad_norm=rand_grad_norms[2*i - 1],
                     grad_lr_norm=rand_grad_lr_norms[2*i - 1], w_grad_angle=rand_w_grad_angle[2*i - 1])

name = dir + 'loss_vec_dir.npz'
with open(name, 'wb') as f:
    np.savez(f, loss_vec_dir=loss_vec_dir)

angles = []
for i in range(len(full_grad_dir)):
    angles.append(np.dot(full_grad_dir[i], fin_l_tilde)/(np.linalg.norm(full_grad_dir[i])*np.linalg.norm(fin_l_tilde)))

name = dir + 'l_tilde_grad_angle.npz'
with open(name, 'wb') as f:
    np.savez(f, l_tilde_grad_angle=angles)


