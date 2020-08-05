# encoding=utf8
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available() #and False

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

        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,len), ...]*len(Ks)
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1]  # [(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1)  # (b_s, co*len(Ks))

        x = self.dropout(x3)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, out_dim)

        return logit


class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_size, output_size, bias=True)
        #self.lin2 = nn.Linear(output_size, output_size, bias=True)
        self.lin3 = nn.Linear(output_size, output_size, bias=True)

    def forward(self,x):
        # x: (b_s, len, embsize)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        #x = F.relu(self.lin2(x))
        x2 = self.lin3(x)
        return x2

class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(LSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self,x, x_lengths):
        # x: (b_s, len, embsize)
        #x_lengths = np.array(x_lengths)
        ind1 = torch.argsort(x_lengths,descending=True)
        ind2 = torch.argsort(ind1)

        # decending order
        x = x[ind1]
        x_lengths = x_lengths[ind1]

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        #print 'X: ',X
        outputs,_ = self.lstm_layer(X)

        # undo the packing operation
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # recover the order
        outputs = outputs[ind2]

        #print 'outputs: ',outputs.size()
        out = outputs[:,-1,:]

        return out # Obtaining the last output



def package_qd(qids, dids, query_var, doc_var, q_lens, d_lens):
    id_list, text_list, len_list, qd_mask = [],[],[],[]

    for i,id_ in enumerate(qids):
        if id_ not in id_list:
            id_list.append(id_)
            text_list.append(query_var[i])
            len_list.append(q_lens[i])
            qd_mask.append()

    for i,id_ in enumerate(dids):
        if id_ not in id_list:
            id_list.append(id_)
            text_list.append(doc_var[i])
            len_list.append(d_lens[i])

    return id_list, text_list, len_list, qd_mask


def exclusive_combine(*in_list):
    res = set()
    in_list = list(*in_list)
    for n_l in in_list:
        for i in n_l:
            res.add(i)
    return list(res)

def pad_seq(seq, max_length, PAD_token=0):
    seq = seq[:max_length]
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def pad_seq_bert(seq, max_length, PAD_token=103):
    #id of [MASK] is 103
    if len(seq) > max_length:
        seq = seq[:max_length - 1] + [seq[-1]]
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def Tensor2Varible(tensor_):
    var = Variable(tensor_)
    var = var.cuda() if use_cuda else var
    return var

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