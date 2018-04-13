import numpy as np
import torch
import torch.nn as nn

class Binop:
    def __init__(self,model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.bin_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
    
    def ClampWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp(-1.0,1.0,out=self.target_modules[index].data)
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def BinarizeWeights(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                alpha = self.target_modules[index].data.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n)
            elif len(s) == 2:
                alpha = self.target_modules[index].data.norm(1,1,keepdim=True).div(n)
            self.target_modules[index].data.sign().mul(alpha.expand(s),out=self.target_modules[index].data)
    
    def Binarization(self):
        self.ClampWeights()
        self.SaveWeights()
        self.BinarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    
    def UpdateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                alpha = weight.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                alpha = weight.norm(1,1,keepdim=True).div(n).expand(s)
            alpha[weight.le(-1.0)] = 0
            alpha[weight.ge(1.0)] = 0
            alpha = alpha.mul(self.target_modules[index].grad.data)
            add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                add = add.sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                add = add.sum(1,keepdim=True).div(n).expand(s)
            add = add.mul(weight.sign())
            self.target_modules[index].grad.data = alpha.add(add)

