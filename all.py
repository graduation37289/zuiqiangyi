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

def sgd_update(parameters, lr):		# 定义SGD优化函数
        for param in parameters:
            param.data = param.data - lr * param.grad.data

train_set = MNIST('./data',	train=True,	transform=data_tf, download=True)  # 加载数据训练集
test_set = MNIST('./data', train=False, transform=data_tf, download=True)  # 加载数据测试集

criterion = nn.CrossEntropyLoss()
train_data = DataLoader(train_set, batch_size=64,	shuffle=True)
net = nn.Sequential(
				nn.Linear(784,	200),
				nn.ReLU(),
				nn.Linear(200,	10),
)

losses1 = []
idx = 0
start = time.time()
for e in range(10):
                    train_loss = 0
                    for im,	label in train_data:
                                    im = Variable(im)
                                    label = Variable(label)

                                    out = net(im)
                                    loss = criterion(out, label)

                                    net.zero_grad()
                                    loss.backward()
                                    sgd_update(net.parameters(), 1e-2)

                                    train_loss += loss.item()
                                    if idx % 300 == 0:
                                        losses1.append(loss.item())
                                    idx += 1
                    print('epoch: {}, Train	Loss: {:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('ֵ使用时间:	{:.5f}	s'.format(end - start))


train_data = DataLoader(train_set, batch_size=64, shuffle=True)
net	= nn.Sequential(
				nn.Linear(784, 200),
				nn.ReLU(),
				nn.Linear(200, 10),
)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

losses2 = []
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
                                    if idx % 300 == 0:
                                        losses2.append(loss.item())
                                    idx += 1
                    print('epoch: {}, Train	Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))



def adagrad(parameters, sqrs, lr):	# 定义adagrad优化算法
        eps = 1e-10
        for param, sqr in zip(parameters, sqrs):
            sqr[:] = sqr + param.grad.data ** 2
            div = lr / torch.sqrt(sqr + eps) * param.grad.data
            param.data = param.data - div


train_data = DataLoader(train_set, batch_size=64, shuffle=True)

net = nn.Sequential(
				nn.Linear(784,	200),
				nn.ReLU(),
				nn.Linear(200,	10),
)
optimizer = torch.optim.Adagrad(net.parameters(), lr=1e-2)

sqrs = []
for param in net.parameters():
    sqrs.append(torch.zeros_like(param.data))

losses3 = []
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
                        adagrad(net.parameters(), sqrs, 1e-2)
                        train_loss += loss.item()
                        if idx % 300 == 0:
                            losses3.append(loss.item())
                        idx += 1
            print('epoch: {}, Train Loss: {:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))


def rmsprop(parameters, sqrs, lr, alpha):  # 定义rmsprop优化函数
    eps = 1e-10
    for param, sqr in zip(parameters, sqrs):
        sqr[:] = alpha * sqr + (1 - alpha) * param.grad.data ** 2
        div = lr / torch.sqrt(sqr + eps) * param.grad.data
        param.data = param.data - div

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)
sqrs = []
for param in net.parameters():
    sqrs.append(torch.zeros_like(param.data))
losses4 = []
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
            if idx % 300 == 0:
                losses4.append(loss.item())
            idx += 1
    print('epoch:{}, Train	Loss:{:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('使用时间:{:.5f} s'.format(end - start))


def adadelta(parameters, sqrs, deltas, rho):
    eps = 1e-6
    for param, sqr, delta in zip(parameters, sqrs, deltas):
        sqr[:] = rho * sqr + (1 - rho) * param.grad.data ** 2
        cur_delta = torch.sqrt(delta + eps) / torch.sqrt(sqr + eps) * param.grad.data
        delta[:] = rho * delta + (1 - rho) * cur_delta ** 2
        param.data = param.data - cur_delta

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
losses5 = []
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
                                    if idx % 300 == 0:
                                                losses5.append(loss.item())
                                    idx	+= 1
                    print('epoch: {}, Train	Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))


def adam(parameters, vs, sqrs, lr, t, beta1=0.9, beta2=0.999):
    eps = 1e-8
    for param, v, sqr in zip(parameters, vs, sqrs):
        v[:] = beta1 * v + (1 - beta1) * param.grad.data
        sqr[:] = beta2 * sqr + (1 - beta2) * param.grad.data ** 2
        v_hat = v / (1 - beta1 ** t)
        s_hat = sqr / (1 - beta2 ** t)
        param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)

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
losses6 = []
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
            if idx % 300 == 0:
                losses6.append(loss.item())
            idx += 1
    print('epoch:{}, Train	Loss:{:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('使用时间:{:.5f} s'.format(end - start))


x_axis = np.linspace(0,	10,	len(losses2), endpoint=True)
plt.semilogy(x_axis, losses1, label='SGD_batch_size=64')
plt.semilogy(x_axis, losses2, label='动量法momentum: 0.9')
plt.semilogy(x_axis, losses3, label='adagrad')
plt.semilogy(x_axis, losses4, label='RMSProp_alpha=0.999')
plt.semilogy(x_axis, losses5, label='adadelta_rho=0.9')
plt.semilogy(x_axis, losses6, label='adam')
plt.legend(loc='best')
plt.show()
