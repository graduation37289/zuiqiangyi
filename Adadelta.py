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


def adadelta(parameters, sqrs, deltas,	rho):
                    eps	=	1e-6
                    for param, sqr, delta in zip(parameters, sqrs, deltas):
                              sqr[:] = rho * sqr + (1 - rho) * param.grad.data ** 2
                              cur_delta	= torch.sqrt(delta + eps) / torch.sqrt(sqr + eps) * param.grad.data
                              delta[:] = rho * delta + (1 - rho) * cur_delta ** 2
                              param.data = param.data - cur_delta


train_set = MNIST('./data',	train=True,	transform=data_tf, download=True)
test_set = MNIST('./data', train=False, transform=data_tf, download=True)
criterion = nn.CrossEntropyLoss()


# adadelta优化方法rho=0.9和rho=0.99比较
train_data = DataLoader(train_set,	batch_size=64,	shuffle=True)
net = nn.Sequential(
        nn.Linear(784,	200),
        nn.ReLU(),
        nn.Linear(200,	10),
)
sqrs = []
deltas = []
for param in net.parameters():
        sqrs.append(torch.zeros_like(param.data))
        deltas.append(torch.zeros_like(param.data))
losses = []
idx = 0
start = time.time()
for e in range(10):
                    train_loss = 0
                    for im,	label in train_data:
                                    im = Variable(im)
                                    label = Variable(label)
                                    out	= net(im)
                                    loss = criterion(out, label)
                                    net.zero_grad()
                                    loss.backward()
                                    adadelta(net.parameters(), sqrs, deltas, 0.9)
                                    train_loss += loss.item()
                                    if idx % 100 == 0:
                                                losses.append(loss.item())
                                    idx	+= 1
                    print('epoch: {}, Train	Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))


net = nn.Sequential(
        nn.Linear(784,	200),
        nn.ReLU(),
        nn.Linear(200,	10),
)
sqrs = []
deltas = []
for param in net.parameters():
        sqrs.append(torch.zeros_like(param.data))
        deltas.append(torch.zeros_like(param.data))
losses1 = []
idx = 0
start = time.time()
for e in range(10):
                    train_loss = 0
                    for im,	label in train_data:
                                    im = Variable(im)
                                    label = Variable(label)
                                    out	= net(im)
                                    loss = criterion(out, label)
                                    net.zero_grad()
                                    loss.backward()
                                    adadelta(net.parameters(), sqrs, deltas, 0.9)
                                    train_loss += loss.item()
                                    if idx % 100 == 0:
                                                losses1.append(loss.item())
                                    idx	+= 1
                    print('epoch: {}, Train	Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))


# net = nn.Sequential(
#         nn.Linear(784,	200),
#         nn.ReLU(),
#         nn.Linear(200,	10),
# )
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
#
# losses1 = []
# idx = 0
# start = time.time()
# for e in range(10):
#                     train_loss = 0
#                     for im,	label in train_data:
#                                 im = Variable(im)
#                                 label = Variable(label)
#                                 out = net(im)
#                                 loss = criterion(out, label)
#                                 optimizer.zero_grad()
#                                 loss.backward()
#                                 optimizer.step()
#                                 train_loss += loss.item()
#                                 if idx % 100 == 0:
#                                             losses1.append(loss.item())
#                                 idx += 1
#                     print('epoch: {}, Train Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
# end = time.time()
# print('ֵ使用时间: {:.5f} s'.format(end - start))

x_axis = np.linspace(0,	10,	len(losses), endpoint=True)
plt.semilogy(x_axis, losses, label='rho=0.9')
plt.semilogy(x_axis, losses1, label='rho=0.99')
plt.legend(loc='best')
plt.show()
