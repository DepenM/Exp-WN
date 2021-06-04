import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import seaborn as sns
import os
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

d= 32*32*3
k = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_length = 50000
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
test_length = 10000
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

num_conv_layers = 13
num_fc_layers = 1
kernel_size = [3,3,3,3,3,3,3,3,3,3,3,3,3]
num_filters = [64,64,128,128,256,256,256,512,512,512,512,512,512]
max_pool_indices = [1,3,6,9,12]
fc_neurons = []
log_loss_limit = 0
save_values = [1, 0]
curr_save_index = 0
norm_type = args.type
if norm_type == 'EWN':
    WN = True
    exp_WN = True
    dir = './CIFAR_pruning/' + str(seed) + '/EWN/'
elif norm_type == 'SWN':
    WN = True
    exp_WN = False
    dir = './CIFAR_pruning/' + str(seed) + '/SWN/'
else:
    WN = False
    exp_WN = False
    dir = './CIFAR_pruning/' + str(seed) + '/no-WN/'

CHECK_FOLDER = os.path.isdir(dir)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(dir)

act = 'relu'
lr = 0.01
r_u = 1.1
r_d = 1.1
tot_epochs = 10000

weight_layer_norm = []
last_log_loss = torch.tensor(0.0)

class TestNet(nn.Module):
    def __init__(self, k, num_conv_layers, num_fc_layers, kernel_size, num_filters, fc_neurons, act, max_pool_indices):
        super(TestNet, self).__init__()
        if WN:
            self.conv_w = nn.ParameterList()
            self.conv_s = nn.ParameterList()
            self.w = nn.ParameterList()
            self.s = nn.ParameterList()
        else:
            self.conv = nn.ModuleList()
            self.linears = nn.ModuleList()

        self.max_pool_indices = max_pool_indices
        fin_linear_size = int(num_filters[-1]*((32/np.power(2, len(max_pool_indices)))**2))
        for i in range(num_conv_layers):
            if i==0:
                if WN:
                    temp_param = torch.randn(num_filters[0], 3, kernel_size[0], kernel_size[0])
                    temp_2 = temp_param.view(num_filters[0], -1)
                    temp_2 = torch.reshape(torch.norm(temp_2, dim=1, keepdim=True), (temp_2.shape[0],1,1,1))
                    self.conv_w.append(nn.Parameter(temp_param/temp_2))
                    if exp_WN:
                        self.conv_s.append(nn.Parameter(torch.zeros(num_filters[0], 1)))
                    else:
                        self.conv_s.append(nn.Parameter(torch.ones(num_filters[0], 1)))
                    self.conv_b = nn.Parameter(torch.zeros(num_filters[0]))
                else:
                    self.conv.append(nn.Conv2d(3, num_filters[0], kernel_size[0], bias=True, padding=1))
            else:
                if WN:
                    temp_param = torch.randn(num_filters[i], num_filters[i-1], kernel_size[i], kernel_size[i])
                    temp_2 = temp_param.view(num_filters[i], -1)
                    temp_2 = torch.reshape(torch.norm(temp_2, dim=1, keepdim=True), (temp_2.shape[0], 1, 1, 1))
                    self.conv_w.append(nn.Parameter(temp_param/temp_2))
                    if exp_WN:
                        self.conv_s.append(nn.Parameter(torch.zeros(num_filters[i], 1)))
                    else:
                        self.conv_s.append(nn.Parameter(torch.ones(num_filters[i], 1)))
                else:
                    self.conv.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size[i], bias=False, padding=1))

        if num_fc_layers==1:
            if WN:
                self.lin = nn.Linear(fin_linear_size, k, bias=False)
            else:
                self.linears.append(nn.Linear(fin_linear_size, k, bias=False))
        else:
            for i in range(num_fc_layers):
                if i==0:
                    if WN:
                        temp_param = torch.randn(fin_linear_size, fc_neurons[0])
                        self.w.append(nn.Parameter(temp_param/torch.norm(temp_param, dim=0, keepdim=True)))
                        if exp_WN:
                            self.s.append(nn.Parameter(torch.zeros(1, fc_neurons[0])))
                        else:
                            self.s.append(nn.Parameter(torch.ones(1, fc_neurons[0])))
                    else:
                        self.linears.append(nn.Linear(fin_linear_size, fc_neurons[0], bias=False))
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
            self.conv.apply(self.w_init)
        self.conv_layers = num_conv_layers
        self.fc_layers = num_fc_layers
        self.act = act

    def forward(self, x):
        for i in range(self.conv_layers):
            if WN:
                w0 = self.conv_w[i]
                w1 = w0.view(w0.shape[0],-1)
                if exp_WN:
                    w2 = torch.reshape(torch.exp(self.conv_s[i])/torch.norm(w1, dim=1, keepdim=True), (w1.shape[0],1,1,1))*w0
                else:
                    w2 = torch.reshape(self.conv_s[i]/torch.norm(w1, dim=1, keepdim=True), (w1.shape[0],1,1,1)) * w0

                if i==0:
                    x = nn.functional.conv2d(x, w2, bias=self.conv_b, padding=1)
                else:
                    x = nn.functional.conv2d(x, w2, padding=1)

                if self.act!='linear':
                    if self.act=='relu':
                        x = F.relu(x)
                    elif self.act=='hardtanh':
                        x = F.hardtanh(x)

                if i in self.max_pool_indices:
                    x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            else:
                x = self.conv[i](x)
                if self.act != 'linear':
                    if self.act == 'relu':
                        x = F.relu(x)
                    elif self.act == 'hardtanh':
                        x = F.hardtanh(x)

                if i in self.max_pool_indices:
                    x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.shape[0],-1)
        for i in range(self.fc_layers):
            if WN:
                if i==self.fc_layers - 1:
                    x = self.lin(x)
                else:
                    w1 = self.w[i]
                    if exp_WN:
                        w1 = (torch.exp(self.s[i]) / torch.norm(w1, dim=0, keepdim=True)) * w1
                    else:
                        w1 = (self.s[i] / torch.norm(w1, dim=0, keepdim=True)) * w1

                    x = torch.matmul(x, w1)
                    if i != self.fc_layers - 1 and self.act != 'linear':
                        if self.act == 'relu':
                            x = F.relu(x)
                        elif self.act == 'hardtanh':
                            x = F.hardtanh(x)
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
        elif type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    if WN:
        WN = False
        net = TestNet(k, num_conv_layers, num_fc_layers, kernel_size, num_filters, fc_neurons, act, max_pool_indices)
        net.to(device)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            break
        conv_params = []
        lin_params = []
        for name, param in net.named_parameters():
            if 'conv' in name:
                if 'bias' not in name:
                    conv_params.append(param)
            else:
                lin_params.append(param)
        WN = True
        net = TestNet(k, num_conv_layers, num_fc_layers, kernel_size, num_filters, fc_neurons, act, max_pool_indices)
        net.to(device)
        state_dict = net.state_dict()
        for name, param in net.named_parameters():
            if 'conv' in name:
                if 's' in name:
                    ind = int(name.split('.')[1])
                    temp = conv_params[ind].view(conv_params[ind].shape[0], -1)
                    if exp_WN:
                        state_dict[name].copy_(torch.log(torch.norm(temp, dim=1, keepdim=True)))
                    else:
                        state_dict[name].copy_(
                            torch.norm(temp, dim=1, keepdim=True))
                elif 'w' in name:
                    ind = int(name.split('.')[1])
                    temp = conv_params[ind].view(conv_params[ind].shape[0], -1)
                    temp = torch.reshape(torch.norm(temp, dim=1, keepdim=True), (temp.shape[0], 1, 1, 1))
                    state_dict[name].copy_(conv_params[ind] / temp)
            else:
                if 'lin' in name:
                    state_dict[name].copy_(lin_params[-1])
                elif 's' in name:
                    ind = int(name.split('.')[1])
                    if exp_WN:
                        state_dict[name].copy_(
                            torch.log(torch.norm(lin_params[ind], dim=1, keepdim=True).transpose(0, 1)))
                    else:
                        state_dict[name].copy_(
                            torch.norm(lin_params[ind], dim=1, keepdim=True).transpose(0, 1))
                else:
                    ind = int(name.split('.')[1])
                    state_dict[name].copy_(
                        (lin_params[ind] / torch.norm(lin_params[ind], dim=1, keepdim=True)).transpose(0, 1))

        outputs2 = net(inputs)
        #print(torch.norm(outputs - outputs2))
    else:
        net = TestNet(k, num_conv_layers, num_fc_layers, kernel_size, num_filters, fc_neurons, act, max_pool_indices)
        net.to(device)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    criterion = torch.nn.functional.cross_entropy

    optimizer = optim.SGD(net.parameters(), lr)

    temp_vec = torch.zeros(10, k - 1).type(torch.LongTensor).to(device)
    for j in range(10):
        temp_index = 0
        for l in range(k):
            if l == j:
                continue
            else:
                temp_vec[j][temp_index] = l
                temp_index += 1

    best_val_acc = 0
    epoch = 0

    while(True):  # loop over the dataset multiple times
        net = net.train()
        if  epoch==0:
            total_loss = 0.0
            acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    class_pred = torch.argmax(outputs, dim=1)
                    acc = acc + torch.sum(class_pred == labels)
                    total_loss += loss.item()*inputs.shape[0]
            train_loss.append(np.log(total_loss / train_length))
            # print('\n\n')
            print('[%d] Train loss: %.3f, Train acc: %.3f' %
                  (epoch + 1, total_loss / train_length, acc.item()/train_length))

            last_log_loss = torch.tensor(np.log(total_loss / train_length))

            net = net.eval()
            total_loss = 0.0
            acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    total_loss += loss.item()*inputs.shape[0]
                    class_pred = torch.argmax(outputs, dim=1)
                    acc = acc + torch.sum(class_pred==labels)

            print('[%d] Validation loss: %.3f, Validation acc: %.3f' %
                  (epoch + 1, total_loss / test_length, acc.item()/test_length))

            test_loss.append(total_loss / test_length)
            net = net.train()

        running_loss = 0.0
        acc = 0.0
        #save parameters so as to restore if loss value goes up
        temp_state = {}
        for key in net.state_dict().keys():
            temp_state[key] = net.state_dict()[key].clone().detach()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            temp = outputs.gather(1,labels.unsqueeze(1))
            outputs2 = temp - outputs
            temp_vec_2 = temp_vec[labels]
            s_n = outputs2.gather(1, temp_vec_2)
            s_n_min = None
            if torch.min(s_n) > 0:
                s_n_min = torch.min(s_n)
            if torch.min(s_n) > 30:
                loss = torch.mean(torch.exp(torch.logsumexp(-s_n, dim=1, keepdim=True) - last_log_loss))
            else:
                loss = torch.mean(torch.log1p(torch.sum(torch.exp(-s_n), dim=1))*torch.exp(-last_log_loss))

            class_pred = torch.argmax(outputs, dim=1)
            acc = acc + torch.sum(class_pred == labels)

            loss.backward()
            if i == 0:  # and epoch%10==0:
                if WN:
                    conv_s = []
                    lin_s = []
                    for name, param in net.named_parameters():
                        if 'conv' in name:
                            if 's' in name:
                                conv_s.append(param)
                        else:
                            if 'lin' in name:
                                last_weight = param
                            if 's' in name:
                                lin_s.append(param)

                    for j in range(num_conv_layers):
                        if exp_WN:
                            if len(weight_layer_norm) <= j:
                                weight_layer_norm.append(torch.exp(conv_s[j]).detach().cpu().numpy())
                            else:
                                weight_layer_norm[j] = np.concatenate([weight_layer_norm[j],
                                                                       torch.exp(conv_s[j]).detach().cpu().numpy()],
                                                                      axis=1)
                        else:
                            if len(weight_layer_norm) <= j:
                                weight_layer_norm.append(torch.abs(conv_s[j]).detach().cpu().numpy())
                            else:
                                weight_layer_norm[j] = np.concatenate([weight_layer_norm[j],
                                                                       torch.abs(conv_s[j]).detach().cpu().numpy()],
                                                                      axis=1)

                    for j in range(num_fc_layers):
                        if j == num_fc_layers - 1:
                            if len(weight_layer_norm) <= num_conv_layers + j:
                                weight_layer_norm.append(
                                    torch.norm(last_weight, dim=1, keepdim=True).transpose(0, 1).detach().cpu().numpy())
                            else:
                                weight_layer_norm[num_conv_layers + j] = np.concatenate(
                                    [weight_layer_norm[num_conv_layers + j],
                                     torch.norm(last_weight, dim=1,
                                                keepdim=True).transpose(0, 1).detach().cpu().numpy()],
                                    axis=0)
                        elif exp_WN:
                            if len(weight_layer_norm) <= num_conv_layers + j:
                                weight_layer_norm.append(torch.exp(lin_s[j]).detach().cpu().numpy())
                            else:
                                weight_layer_norm[num_conv_layers + j] = np.concatenate(
                                    [weight_layer_norm[num_conv_layers + j],
                                     torch.exp(lin_s[j]).detach().cpu().numpy()],
                                    axis=0)
                        else:
                            if len(weight_layer_norm) <= num_conv_layers + j:
                                weight_layer_norm.append(torch.abs(lin_s[j]).detach().cpu().numpy())
                            else:
                                weight_layer_norm[num_conv_layers + j] = np.concatenate(
                                    [weight_layer_norm[num_conv_layers + j],
                                     torch.abs(lin_s[j]).detach().cpu().numpy()],
                                    axis=0)
                else:
                    for name, param in net.named_parameters():
                        ind = int(name.split('.')[1])
                        if 'conv' in name:
                            if 'weight' in name:
                                temp = param.view(param.shape[0], -1)
                                if len(weight_layer_norm) <= ind:
                                    weight_layer_norm.append(
                                        torch.norm(temp, dim=1, keepdim=True).detach().cpu().numpy())
                                else:
                                    weight_layer_norm[ind] = np.concatenate([weight_layer_norm[ind],
                                                                             torch.norm(temp, dim=1, keepdim=True)
                                                                            .detach().cpu().numpy()],
                                                                            axis=1)
                        else:
                            if len(weight_layer_norm) <= num_conv_layers + ind:
                                weight_layer_norm.append(torch.norm(param, dim=1, keepdim=True).detach().cpu().numpy())
                            else:
                                weight_layer_norm[num_conv_layers + ind] = np.concatenate(
                                    [weight_layer_norm[num_conv_layers + ind],
                                     torch.norm(param, dim=1, keepdim=True)
                                    .detach().cpu().numpy()],
                                    axis=1)

            optimizer.step()

            running_loss += loss.item()*inputs.shape[0]
            if running_loss != running_loss:
                break

        #loss value can be nan due to numerical underflow
        if running_loss/train_length != running_loss/train_length:
            lr = lr / r_d
            for g in optimizer.param_groups:
                g['lr'] = lr
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key].copy_(temp_state[key])
            continue

        with torch.no_grad():
            acc = 0.0
            running_loss = 0.0
            for i,data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
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
                running_loss += loss.item()*inputs.shape[0]

        if running_loss/train_length > 1 or running_loss/train_length != running_loss/train_length:
            #decrease lr if loss value goes up
            lr = lr/r_d
            for g in optimizer.param_groups:
                g['lr'] = lr
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key].copy_(temp_state[key])
            continue
        else:
            #increase lr if loss value goes down
            lr = lr*r_u
            for g in optimizer.param_groups:
                g['lr'] = lr
            last_log_loss = last_log_loss + torch.tensor(np.log(running_loss/train_length))
            epoch += 1

        train_loss.append(last_log_loss)
        train_acc.append(acc.item()/train_length)

        print('[%d] Train loss: %.3f, Train acc: %.3f' %
              (epoch + 1, last_log_loss, acc.item()/train_length))

        net = net.eval()
        acc = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                class_pred = torch.argmax(outputs, dim=1)
                acc = acc + torch.sum(class_pred == labels)
        print('[%d] Validation loss: %.3f, Validation acc: %.3f' %
              (epoch + 1, total_loss / (i + 1), acc.item()/test_length))

        test_loss.append(total_loss / (i + 1))
        test_acc.append(acc.item()/test_length)

        if last_log_loss < save_values[curr_save_index]:
            dir2 = dir + str(save_values[curr_save_index])

            CHECK_FOLDER = os.path.isdir(dir2)

            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(dir2)

            torch.save(net.state_dict(), dir2 + '/weights.pth')

            for i in range(len(weight_layer_norm)):
                name = dir2 + '/weight_norm_' + str(i) + '.npz'
                with open(name, 'wb') as f:
                    np.savez(f, w_norm=weight_layer_norm[i])

            str_train_loss = []
            for i in range(len(train_loss)):
                str_train_loss.append(str(train_loss[i]))
            name = dir2 + '/train_loss.txt'
            with open(name, 'w') as f:
                f.write('\n'.join(str_train_loss))
            f.close()

            str_test_loss = []
            for i in range(len(test_loss)):
                str_test_loss.append(str(test_loss[i]))
            name = dir2 + '/test_loss.txt'
            with open(name, 'w') as f:
                f.write('\n'.join(str_test_loss))
            f.close()

            str_train_acc = []
            for i in range(len(train_acc)):
                str_train_acc.append(str(train_acc[i]))
            name = dir2 + '/train_acc.txt'
            with open(name, 'w') as f:
                f.write('\n'.join(str_train_acc))
            f.close()

            str_test_acc = []
            for i in range(len(test_acc)):
                str_test_acc.append(str(test_acc[i]))
            name = dir2 + '/test_acc.txt'
            with open(name, 'w') as f:
                f.write('\n'.join(str_test_acc))
            f.close()

            curr_save_index += 1

        if last_log_loss < -log_loss_limit or lr==0 or epoch>tot_epochs:
            break

        if acc.item()/test_length > best_val_acc:
            best_val_acc = acc.item()/test_length