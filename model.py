import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class BinActiv(Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension
    '''
    @staticmethod
    def forward(ctx,input): #input has already converted to Tensor
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

        self.k = 1 / (kernel_size**2) * torch.ones(-1,1,kernel_size,kernel_size) #constrain kernel as square

    def forward(self,x):
        x = self.bn(x)
        x,A = BinActiv()(x)

        return x



