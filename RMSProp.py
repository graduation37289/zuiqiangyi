import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt


def data_tf(x):
            x = np.array(x,	dtype='float32') / 255
            x = (x - 0.5) / 0.5
            x = x.reshape((-1,))
            x = torch.from_numpy(x)
            return x


def rmsprop(parameters,	sqrs, lr, alpha):   # 定义rmsprop优化函数
                    eps	= 1e-10
                    for param, sqr in zip(parameters,	sqrs):
                              sqr[:] = alpha * sqr + (1 - alpha) * param.grad.data ** 2
                              div = lr / torch.sqrt(sqr + eps) * param.grad.data
                              param.data = param.data - div


train_set = MNIST('./data',	train=True,	transform=data_tf,	download=True)
test_set = MNIST('./data',	train=False,	transform=data_tf,	download=True)
criterion = nn.CrossEntropyLoss()




train_data = DataLoader(train_set, batch_size=64, shuffle=True)
net = nn.Sequential(
              nn.Linear(784, 200),
              nn.ReLU(),
              nn.Linear(200, 10),
)


optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

losses = []
idx = 0
start = time.time()
for e in range(10):
                    train_loss = 0
                    for im,	label in train_data:
                                im = Variable(im)
                                label = Variable(label)
                                out = net(im)
                                loss = criterion(out, label)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                train_loss += loss.item()
                                if idx % 100 == 0:
                                            losses.append(loss.item())
                                idx += 1
                    print('epoch: {}, Train Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('ֵ使用时间: {:.5f} s'.format(end - start))

# rmsprop优化器，lr=0.001,alpha=0.9
# sqrs = []
# for param in net.parameters():
#         sqrs.append(torch.zeros_like(param.data))
# losses = []
# idx = 0
# start = time.time()
# for e in range(10):
#         train_loss = 0
#         for im,	label in train_data:
#                 im = Variable(im)
#                 label = Variable(label)
#                 out = net(im)
#                 loss = criterion(out,	label)
#                 net.zero_grad()
#                 loss.backward()
#                 rmsprop(net.parameters(), sqrs, 1e-3, 0.9)
#                 train_loss += loss.item()
#                 if idx % 100 == 0:
#                         losses.append(loss.item())
#                 idx += 1
#         print('epoch: {}, Train	Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
# end = time.time()
# print('ֵ使用时间: {:.5f} s'.format(end - start))



# rmsprop优化器，lr=0.001,alpha=0.99
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)
sqrs = []
for param in net.parameters():
    sqrs.append(torch.zeros_like(param.data))
losses1 = []
idx = 0
start = time.time()
for e in range(10):
    train_loss = 0
    for im, label in train_data:
            im = Variable(im)
            label = Variable(label)
            out = net(im)
            loss = criterion(out, label)
            net.zero_grad()
            loss.backward()
            rmsprop(net.parameters(), sqrs, 1e-3, 0.999)
            train_loss += loss.item()
            if idx % 100 == 0:
                losses1.append(loss.item())
            idx += 1
    print('epoch:{}, Train	Loss:{:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('使用时间:{:.5f} s'.format(end - start))
x_axis = np.linspace(0, 10, len(losses), endpoint=True)
plt.semilogy(x_axis, losses, label='SGD')
plt.semilogy(x_axis, losses1, label='alpha=0.999')
plt.legend(loc='best')
plt.show()


