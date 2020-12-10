# encoding=utf8
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

def Tensor2Varible(tensor_):
    var = Variable(tensor_)
    var = var.cuda() if use_cuda else var
    return var


class KimCNN(nn.Module):
    def __init__(self, kernel_num, kernel_size, embsize,out_dim, drate=0.1):
        super(KimCNN, self).__init__()
        Ci = 1  # n_in_kernel
        Co = kernel_num
        Ks = range(1,kernel_size+1) #[1,2,...,KernelSize]
        n_out = len(Ks) * Co

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])
        self.dropout = nn.Dropout(drate) #p: probability of an element to be zeroed. Default: 0.5
        self.fc1 = nn.Linear(n_out,out_dim)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        #x: (b_s, len, embsize)
        x = x.unsqueeze(1)  # (N,Ci,len,embsize)
        #print 'x: ',x
        try:
            x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,len), ...]*len(Ks)
            x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1]  # [(N,Co), ...]*len(Ks)
            x3 = torch.cat(x2, 1)  # (b_s, co*len(Ks))
        except Exception as e:
            print e
            print x.size()
            exit(-1)

        x = self.dropout(x3)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, out_dim)

        return logit

def getOptimizer(name,parameters,**kwargs):
    if name == 'sgd':
        return optim.SGD(parameters,**kwargs)
    elif name == 'adadelta':
        return optim.Adadelta(parameters,**kwargs)
    elif name == 'adam':
        return optim.Adam(parameters,**kwargs)
    elif name == 'adagrad':
        return optim.Adagrad(parameters,**kwargs)
    elif name == 'rmsprop':
        return optim.RMSprop(parameters,**kwargs)
    else:
        raise Exception('Optimizer Name Error')
