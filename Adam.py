import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x


def adam(parameters, vs, sqrs, lr, t, beta1=0.9, beta2=0.999):
             eps = 1e-8
             for param,	v,	sqr in zip(parameters,	vs,	sqrs):
                         v[:] = beta1 * v + (1 - beta1)	* param.grad.data
                         sqr[:] = beta2 * sqr + (1 - beta2) * param.grad.data ** 2
                         v_hat = v / (1 - beta1 ** t)
                         s_hat = sqr / (1 - beta2 ** t)
                         param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)


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


train_data = DataLoader(train_set, batch_size=64, shuffle=True)
net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)
sqrs = []
vs = []
for param in net.parameters():
    sqrs.append(torch.zeros_like(param.data))
    vs.append(torch.zeros_like(param.data))
t = 1
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
            adam(net.parameters(), vs, sqrs, 1e-3, t)
            t += 1
            train_loss += loss.item()
            if idx % 100 == 0:
                losses1.append(loss.item())
            idx += 1
    print('epoch:{}, Train	Loss:{:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('使用时间:{:.5f} s'.format(end - start))
x_axis = np.linspace(0, 10, len(losses), endpoint=True)
plt.semilogy(x_axis, losses, label='SGD')
plt.semilogy(x_axis, losses1, label='adam')
plt.legend(loc='best')
plt.show()
