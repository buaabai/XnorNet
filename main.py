import torch
import torch.nn as nn
from torch.autograd import Variable,Function
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import model
import util
import argparse

def parse():
    parser = argparse.ArgumentParser(description='XnorNet Pytorch MNIST Example.')
    parser.add_argument('--batch-size',type=int,default=100,metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=100,metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs',type=int,default=20,metavar='N',
                        help='number of epochs to decay learning rate(default: 20)')
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=1,mevatar='S',
                        help='random seed(default: 1)')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

def save_model(model,acc):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state,'model_state.pkl')
    print('*** DONE! ***')

BATCH_SIZE = 100

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

model = model.LeNet5_Bin()
model.cuda()

criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.Adam(model.parameters())

bin_op = util.Binop(model)

def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()

        bin_op.Binarization()

        output = model(data)
        loss = criterion(output,target)
        loss.backward()

        bin_op.Restore()
        bin_op.UpdateBinaryGradWeight()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    global best_acc
    best_acc = 0.0
    model.eval()
    test_loss = 0
    correct = 0

    bin_op.Binarization()
    for data,target in test_loader:
        data,target = data.cuda(),target.cuda()
        data,target = Variable(data,volatile=True),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).data[0]
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    bin_op.Restore()
    
    acc = 100. * correct/len(test_loader.dataset)
    if(acc > best_acc):
        best_acc = acc

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    #torch.cuda.manual_seed(1)
    for epoch in range(10):
        train(epoch)
        test()
    bin_op.Binarization()
    save_model(model,best_acc)
