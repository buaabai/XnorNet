import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import struct
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 100
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 3)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x),2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x),2))
        x = x.view(-1,400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  

def accuracy(y_pred,y):
    pred = torch.max(y_pred,1)[1]
    correct = (pred == y).long().sum() #需要重点注意
    acc = (correct.data[0] / len(y))
    return acc

model = LeNet5()
model.cuda()

optimizer = optim.Adam(model.parameters())

loss_fn = nn.CrossEntropyLoss()
loss_fn_test = nn.CrossEntropyLoss(size_average=False)
loss_fn_test.cuda()

for epoch in range(100):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        batch_x,batch_y = batch_x.cuda(),batch_y.cuda()
        x = Variable(batch_x)
        y = Variable(batch_y)
        y_pred = model.forward(x)
        loss = loss_fn(y_pred,y)
        train_acc = accuracy(y_pred,y)

        if step % 20 == 0:
            print('Epoch : %d   ||  step = %d   || loss = %f   || train_acc = %f' % (epoch,step,loss.data[0],train_acc))
        
        if step % 100 == 0:
            model.eval()
            test_loss = 0
            test_acc = 0
            for test_x,test_y in test_loader:
                test_x,test_y = test_x.cuda(),test_y.cuda()
                test_x = Variable(test_x,volatile=True)
                test_y = Variable(test_y)
                test_y_pred = model.forward(test_x)
                test_loss += loss_fn_test(test_y_pred,test_y).data[0]
                test_acc += accuracy(test_y_pred,test_y)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            print('******Epoch : %d   ||  step = %d   || test_loss = %f   || test_acc = %f' % (epoch,step,test_loss,test_acc))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
