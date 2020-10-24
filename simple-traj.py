import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
sns.set_style('white')

norm_type = 'no-WN'
inits = [[1.,1.], [3.,1.], [1.,3.], [5.,1.], [1.,5.]]
trajectory = []

for init in inits:
    np.random.seed(2137)
    torch.manual_seed(2137)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #generating training and testing data
    d=2
    k=2
    train_length = 1
    test_length = 1
    training_set = torch.tensor(np.array([[2, 1]])).type(torch.FloatTensor)
    train_labels = torch.tensor(np.array([[0]])).type(torch.FloatTensor)
    val_set = torch.tensor(np.array([[2, 1]])).type(torch.FloatTensor)
    val_labels = torch.tensor(np.array([[0]])).type(torch.FloatTensor)

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

    if norm_type=='exp-WN':
        WN = True
        exp_WN = True
        dir = 'simple-traj/Exp-WN'
    else:
        WN = False
        exp_WN = False
        dir = 'simple-traj/No-WN'

    """
    frac governs the learning rate order
    frac = 10 means lr = 1/(L^(1 - 1/10))
    frac = 0 means lr = 1/L
    """
    frac = 10
    lr = 0.01
    lr_limit = 0.1
    r_u = 1.1
    r_d = 1.1
    batch_size = train_length
    tot_epochs = 100000
    log_loss_limit = 50
    traj_temp = np.reshape(np.array(init), (-1,1))
    last_log_loss = torch.tensor(0.0)

    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            if WN:
                self.w = nn.Parameter(torch.tensor(np.log(np.array(init))))
            else:
                self.w = nn.Parameter(torch.tensor(np.array(init)))

        def forward(self, x):
            if WN:
                x = torch.sum(x*torch.exp(self.w))
            else:
                x = torch.sum(x*self.w)
            return x

    if __name__ == '__main__':
        np.set_printoptions(suppress=True)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                                 num_workers=0)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        net = TestNet()
        net = net.to(device)
        optimizer = optim.SGD(net.parameters(), lr)
        for epoch in range(tot_epochs):  # loop over the dataset multiple times
            net = net.train()
            running_loss = 0.0
            acc = torch.tensor(0.0)
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = torch.sum(torch.exp(-outputs-last_log_loss))
                loss.backward()
                while (True):
                    for g in optimizer.param_groups:
                        if frac == 0:
                            g['lr'] = lr
                        else:
                            g['lr'] = lr * torch.exp(last_log_loss / frac)

                    optimizer.step()

                    new_last_log_loss = torch.log(loss.detach()) + last_log_loss
                    for j, data in enumerate(trainloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = net(inputs)
                    loss_new = torch.sum(torch.exp(-outputs - last_log_loss))
                    loss_new_2 = torch.sum(torch.exp(-outputs - new_last_log_loss))

                    if loss_new > loss or loss_new_2 == 0:
                        for g in optimizer.param_groups:
                            if frac == 0:
                                g['lr'] = -lr
                            else:
                                g['lr'] = -lr * torch.exp(last_log_loss / frac)
                        optimizer.step()
                        lr = lr / r_d
                        if lr == 0:
                            break
                    else:
                        if lr < lr_limit:
                            lr = lr * r_u
                        break

                for name, param in net.named_parameters():
                    if WN:
                        traj_temp = np.concatenate(
                            [traj_temp, np.reshape(torch.exp(param).detach().cpu().numpy(), (-1, 1))], axis=1)
                    else:
                        traj_temp = np.concatenate([traj_temp, np.reshape(param.detach().cpu().numpy(), (-1,1))], axis=1)
                last_log_loss = torch.log(loss.detach()) + last_log_loss
                running_loss += loss.item()
                if i % 500 == 499:
                    t = 1
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / i))

            train_loss.append(last_log_loss.detach().cpu().numpy())
            train_acc.append(acc.item()/train_length)

            print('[%d] Train loss: %.30f, Train acc: %.3f' %
                  (epoch + 1, last_log_loss.detach().cpu().numpy(), acc.item()/train_length))

            net = net.eval()
            acc = 0.0
            total_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)

                    loss = torch.sum(torch.exp(-outputs))
                    total_loss += loss.item()
                    class_pred = outputs > 0
                    acc = acc + torch.sum(class_pred == labels)
            print('[%d] Validation loss: %.3f, Validation acc: %.3f' %
                  (epoch + 1, total_loss/test_length, acc.item()/test_length))

            test_loss.append(total_loss / test_length)
            test_acc.append(acc.item()/test_length)

            if last_log_loss < -log_loss_limit or lr==0:
                break

    torch.save(net.state_dict(), dir + '/weights.pth')

    name = dir + '/fin_log_loss.txt'
    with open(name, 'w') as f:
        f.write(str(last_log_loss))
    f.close()

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

    trajectory.append(traj_temp)

for i in range(len(trajectory)):
    name = dir + '/traj_' + str(i) + '.npz'
    with open(name, 'wb') as f:
        np.savez(f, trajectory=trajectory[i])

