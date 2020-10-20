import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import seaborn as sns
sns.set_style('white')

seed = 9192
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
log_loss_limit = 10
norm_type = 'No-WN'
if norm_type == 'exp-WN':
    WN = True
    exp_WN = True
    dir = 'MNIST_pruning/Exp-WN/e-' + str(log_loss_limit)
elif norm_type == 'Standard-WN':
    WN = True
    exp_WN = False
    dir = 'MNIST_pruning/Standard-WN/e-' + str(log_loss_limit)
else:
    WN = False
    exp_WN = False
    dir = 'MNIST_pruning/No-WN/e-' + str(log_loss_limit)

act = 'relu'
lr = 0.01
r_u = 1.1
r_d = 1.1
tot_epochs = 10000

#set load=True and sppecify load dir if want to resume training from some last loss value
load = False
if load:
    load_dir = ''

weight_layer_norm = []
last_log_loss = torch.tensor(0.0)

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
    if load:
        net = TestNet(d, k, num_fc_layers, fc_neurons, act)
        net.to(device)
        net.load_state_dict(torch.load(load_dir + '/weights.pth'))
        filename = load_dir + '/fin_log_loss.txt'
        with open(filename, 'r') as f:
            last_log_loss = torch.tensor(float(f.readline()))
        filename = load_dir + '/fin_lr.txt'
        with open(filename, 'r') as f:
            lr = float(f.readline())
    elif WN:
        WN = False
        net = TestNet(d, k, num_fc_layers, fc_neurons, act)
        net.to(device)
        name2 = dir + '/init_weight_layer_norm.txt'
        init_weight_layer_norm = []
        for name, param in net.named_parameters():
            if 'bias' not in name:
                init_weight_layer_norm.append(torch.norm(param, dim=1, keepdim=True).detach().cpu().numpy())
        for i in range(len(init_weight_layer_norm)):
            str_w_norm = []
            for j in range(init_weight_layer_norm[i].shape[0]):
                str_w_norm.append(str(init_weight_layer_norm[i][j, -1]))
            with open(name2, 'a+') as f:
                f.write('\n'.join(str_w_norm))
                f.write('\n\n')
        f.close()

        for i, data in enumerate(trainloader, 0):
            inputs,labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            break
        lin_params = []
        for name, param in net.named_parameters():
            if 'bias' not in name:
                lin_params.append(param)
        WN = True
        net = TestNet(d, k, num_fc_layers, fc_neurons, act)
        net.to(device)
        state_dict = net.state_dict()
        for name, param in net.named_parameters():
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
            elif 'w' in name:
                ind = int(name.split('.')[1])
                state_dict[name].copy_(
                    (lin_params[ind] / torch.norm(lin_params[ind], dim=1, keepdim=True)).transpose(0, 1))

        outputs2 = net(inputs)
        #print(torch.norm(outputs-outputs2))
    else:
        net = TestNet(d, k, num_fc_layers, fc_neurons, act)
        net.to(device)
        name2 = dir + '/init_weight_layer_norm.txt'
        init_weight_layer_norm = []
        for name, param in net.named_parameters():
            if 'bias' not in name:
                init_weight_layer_norm.append(torch.norm(param, dim=1, keepdim=True).detach().cpu().numpy())
        for i in range(len(init_weight_layer_norm)):
            str_w_norm = []
            for j in range(init_weight_layer_norm[i].shape[0]):
                str_w_norm.append(str(init_weight_layer_norm[i][j, -1]))
            with open(name2, 'a+') as f:
                f.write('\n'.join(str_w_norm))
                f.write('\n\n')
        f.close()

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
        if not load and epoch==0:
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
            if i==0:
                if WN:
                    lin_s = []
                    for name, param in net.named_parameters():
                        if 'lin' in name:
                            last_weight = param
                        elif 's' in name:
                            lin_s.append(param)

                    for j in range(num_fc_layers):
                        if j==num_fc_layers-1:
                            if len(weight_layer_norm) <= j:
                                weight_layer_norm.append(torch.norm(last_weight, dim=1, keepdim=True).transpose(0,1).detach().cpu().numpy())
                            else:
                                weight_layer_norm[j] = np.concatenate([weight_layer_norm[j],
                                                                       torch.norm(last_weight, dim=1, keepdim=True).transpose(0,1).detach().cpu().numpy()],
                                                                      axis=0)
                        elif exp_WN:
                            if len(weight_layer_norm) <= j:
                                weight_layer_norm.append(torch.exp(lin_s[j]).detach().cpu().numpy())
                            else:
                                weight_layer_norm[j] = np.concatenate([weight_layer_norm[j],
                                                                       torch.exp(lin_s[j]).detach().cpu().numpy()],
                                                                      axis=0)
                        else:
                            if len(weight_layer_norm) <= j:
                                weight_layer_norm.append(torch.abs(lin_s[j]).detach().cpu().numpy())
                            else:
                                weight_layer_norm[j] = np.concatenate([weight_layer_norm[j],
                                                                       torch.abs(lin_s[j]).detach().cpu().numpy()],
                                                                      axis=0)
                else:
                    for name, param in net.named_parameters():
                        ind = int(name.split('.')[1])
                        if 'bias' not in name:
                            if len(weight_layer_norm) <= ind:
                                weight_layer_norm.append(torch.norm(param, dim=1, keepdim=True).detach().cpu().numpy())
                            else:
                                weight_layer_norm[ind] = np.concatenate([weight_layer_norm[ind],
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

        if last_log_loss < -log_loss_limit or lr==0 or epoch>tot_epochs:
            name = dir + '/fin_log_loss.txt'
            with open(name, 'w') as f:
                f.write(str(last_log_loss.detach().cpu().numpy()))
            f.close()
            name = dir + '/fin_lr.txt'
            with open(name, 'w') as f:
                f.write(str(lr))
            f.close()
            break

        if acc.item()/test_length > best_val_acc:
            best_val_acc = acc.item()/test_length

torch.save(net.state_dict(), dir + '/weights.pth')
name = dir + '/weight_layer_norm.txt'
for i in range(len(weight_layer_norm)):
    str_w_norm = []
    if WN:
        for j in range(weight_layer_norm[i].shape[1]):
            str_w_norm.append(str(weight_layer_norm[i][-1,j]))
    else:
        for j in range(weight_layer_norm[i].shape[0]):
            str_w_norm.append(str(weight_layer_norm[i][j, -1]))
    with open(name, 'a+') as f:
        f.write('\n'.join(str_w_norm))
        f.write('\n\n')
f.close()

for i in range(len(weight_layer_norm)):
    if WN:
        for j in range(weight_layer_norm[i].shape[1]):
            plt.plot(np.arange(weight_layer_norm[i].shape[0]), weight_layer_norm[i][:,j], label=str(j))
    else:
        for j in range(weight_layer_norm[i].shape[0]):
            plt.plot(np.arange(weight_layer_norm[i].shape[1]), weight_layer_norm[i][j, :], label=str(j))
    plt.legend()
    plt.savefig(dir + '/weight_layer_' + str(i) + '.png')
    plt.close()

for i in range(len(weight_layer_norm)):
    name = dir + '/weight_norm_' + str(i) + '.npz'
    with open(name, 'wb') as f:
        np.savez(f, w_norm=weight_layer_norm[i])

print('Best_val_acc is ' + str(best_val_acc))

plt.plot(list(range(len(train_loss))), train_loss, label='train_loss')
plt.legend(loc='upper right')
plt.show()

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
