import torch
import torch.nn as nn
from torch.autograd import Variable,Function
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import model
import util

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
        correct += pred.eq(target.data.view_as(pred)).cpu.sum()
    
    acc = 100. * correct/len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
    #torch.save(model.state_dict(),'model_params.pkl')
