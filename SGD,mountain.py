import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

def data_tf(x): # 数据预处理，标准化...
				x = np.array(x,	dtype='float32') / 255
				x = (x - 0.5) / 0.5
				x = x.reshape((-1,))
				x = torch.from_numpy(x)
				return x


train_set = MNIST('./data',	train=True,	transform=data_tf, download=True)  # 加载数据训练集
test_set = MNIST('./data', train=False, transform=data_tf, download=True)  # 加载数据测试集

criterion = nn.CrossEntropyLoss()


def sgd_update(parameters, lr):		# 定义SGD优化函数
	for param in parameters:
		param.data = param.data - lr * param.grad.data


# 使用SGD优化器，batch_size=1的情况
# train_data = DataLoader(train_set,	batch_size=1,	shuffle=True) #先令批次为1，看下输出效果
#
# net	= nn.Sequential(
# 				nn.Linear(784,	200),
# 				nn.ReLU(),
# 				nn.Linear(200,	10),
# )
#
# losses1	= []
# idx	= 0
# start = time.time()  # 训练开始时间
# for e in range(10):
# 				train_loss = 0
# 				for im,	label in train_data:
# 								im = Variable(im)
# 								label = Variable(label)
#
# 								out = net(im)
# 								loss = criterion(out,	label)
#
# 								net.zero_grad()
# 								loss.backward()
# 								sgd_update(net.parameters(),	1e-2)
#
# 								train_loss += loss.item()
# 								if idx % 30 == 0:
# 										losses1.append(loss.item())
# 								idx += 1
# 				print('epoch: {}, Train	Loss: {:.6f}'.format(e, train_loss / len(train_data)))
# end = time.time() #训练结束时间
# print('ֵ使用时间:	{:.5f}	s'.format(end - start))
# x_axis = np.linspace(0,	10, len(losses1), endpoint=True)
# plt.semilogy(x_axis, losses1, label='SGD_batch_size=1')
# plt.legend(loc='best')
# plt.show()


# 使用SGD优化器，batch_size=64的情况
# train_data = DataLoader(train_set, batch_size=64,	shuffle=True)
#
# net = nn.Sequential(
# 				nn.Linear(784,	200),
# 				nn.ReLU(),
# 				nn.Linear(200,	10),
# )
#
# losses2 = []
# idx = 0
# start = time.time()
# for e in range(10):
# 				train_loss = 0
# 				for im,	label in train_data:
# 								im = Variable(im)
# 								label = Variable(label)
#
# 								out = net(im)
# 								loss = criterion(out, label)
#
# 								net.zero_grad()
# 								loss.backward()
# 								sgd_update(net.parameters(), 1e-2)
#
# 								train_loss += loss.item()
# 								if idx % 30 == 0:
# 									losses2.append(loss.item())
# 								idx += 1
# 				print('epoch: {}, Train	Loss: {:.6f}'.format(e, train_loss / len(train_data)))
# end = time.time()
# print('ֵ使用时间:	{:.5f}	s'.format(end - start))
#
# x_axis = np.linspace(0,	10,	len(losses2), endpoint=True)
# print(x_axis)
# print(len(losses2))
# plt.semilogy(x_axis, losses2, label='SGD_batch_size=64')
# plt.legend(loc='best')
# plt.show()


# 使用SGD优化器，batch_size=64，lr=1的情况
# train_data = DataLoader(train_set, batch_size=64, shuffle=True)
#
# net = nn.Sequential(
# 				nn.Linear(784, 200),
# 				nn.ReLU(),
# 				nn.Linear(200, 10),
# )
#
# losses3 = []
# idx = 0
# start = time.time()
# for e in range(10):
# 				train_loss = 0
# 				for im,	label in train_data:
# 								im = Variable(im)
# 								label = Variable(label)
#
# 								out = net(im)
# 								loss = criterion(out, label)
#
# 								net.zero_grad()
# 								loss.backward()
#
# 								sgd_update(net.parameters(), 1)
#
# 								train_loss += loss.item()
# 								if idx % 30 == 0:
# 									losses3.append(loss.item())
# 								idx += 1
# 				print('epoch: {}, Train	Loss: {:.6f}'.format(e, train_loss / len(train_data)))
# end = time.time()
# print('使用时间:	{:.5f}	s'.format(end - start))
#
# x_axis = np.linspace(0, 10, len(losses3), endpoint=True)
# plt.semilogy(x_axis, losses3, label='SGD_ lr = 1')
# plt.legend(loc='best')
# plt.show()




# 动量法，动量参数为0.9时效果,并与不加动量参数进行比较
train_data = DataLoader(train_set, batch_size=64, shuffle=True)

net	= nn.Sequential(
				nn.Linear(784, 200),
				nn.ReLU(),
				nn.Linear(200, 10),
)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

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
				print('epoch: {}, Train	Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))


net = nn.Sequential(
				nn.Linear(784,	200),
				nn.ReLU(),
				nn.Linear(200,	10),
)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

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
							optimizer.zero_grad()
							loss.backward()
							optimizer.step()
							train_loss += loss.item()
							if idx % 100 == 0:
										losses1.append(loss.item())
							idx += 1
				print('epoch: {}, Train Loss: {:.6f}'.format(e,	train_loss / len(train_data)))
end = time.time()
print('ֵ使用时间: {:.5f} s'.format(end - start))
# x_axis = np.linspace(0,	10, len(losses), endpoint=True)
# plt.semilogy(x_axis, losses, label='momentum: 0.9')
# plt.semilogy(x_axis, losses1, label='no momentum')
# plt.legend(loc='best')
# plt.show()


def adagrad(parameters, sqrs, lr):	# 定义adagrad优化算法
	eps = 1e-10
	for param, sqr in zip(parameters, sqrs):
		sqr[:] = sqr + param.grad.data ** 2
		div = lr / torch.sqrt(sqr + eps) * param.grad.data
		param.data = param.data - div

#adagrad,SGD,动量法比较
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

losses2 = []
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
					if idx % 100 == 0:
						losses2.append(loss.item())
					idx += 1
		print('epoch: {}, Train Loss: {:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print('使用时间: {:.5f} s'.format(end - start))
x_axis = np.linspace(0, 10, len(losses2), endpoint=True)
plt.semilogy(x_axis, losses, label='momentum: 0.9')
plt.semilogy(x_axis, losses1, label='no momentum')
plt.semilogy(x_axis, losses2, label='adagrad')
plt.legend(loc='best')
plt.show()




