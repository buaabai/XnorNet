import torch
import torch.nn as nn
from torch.autograd import Function,Variable
import torch.nn.functional as F

class BinActiv(Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension
    '''
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        mean = torch.mean(input.abs(),1,keepdim=True) #the shape of mnist data is (N,C,W,H)
        input = input.sign()
        return input,mean #tensor

    @staticmethod
    def backward(ctx,grad_output): #grad_output is a Variable
        input,=ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input #Variable

BinActive = BinActiv.apply

class BinConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=False):
        super(BinConv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.layer_type = 'BinConv2d'

        self.bn = nn.BatchNorm2d(in_channels,eps=1e-4,momentum=0.1,affine=True)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,
                            groups=groups,bias=bias)
        self.relu = nn.ReLU()

    def forward(self,x):
        #block structure is BatchNorm -> BinActiv -> BinConv -> Relu
        x = self.bn(x)
        x,A = BinActive(x)
        k = torch.ones(x.shape[0],1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)) #constrain kernel as square
        k = Variable(k)
        K = F.conv2d(A,k,bias=None,stride=self.stride,padding=self.padding,dilation=self.dilation).cuda()
        x = self.conv(x)
        x = x * K
        x = self.relu(x)
        return x

class LeNet5_Bin(nn.Module):
    def __init__(self):
        super(LeNet5_Bin,self).__init__()
        self.conv1 = BinConv2d(1,6,kernel_size = 5)
        self.conv2 = BinConv2d(6,16,kernel_size = 3)
        self.fc1 = nn.Linear(400,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = x.view(-1,400)
        #fc layer still float
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
