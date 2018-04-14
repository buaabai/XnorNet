import torch
import torch.nn as nn
from torch.autograd import Variable,Function
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import model
import util

def save_model(model,acc):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state,'model_state.pkl')
    print('*** DONE! ***')

BATCH_SIZE = 100
learning_rate = 1e-4
weight_decay = 0.001

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
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

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
    
def adjust_learning_rate(optimizer,epoch,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr

if __name__ == '__main__':
    #torch.cuda.manual_seed(1)
    for epoch in range(1000):
        adjust_learning_rate(optimizer,epoch,50)
        train(epoch)
        test()
    bin_op.Binarization()
    save_model(model,best_acc)
