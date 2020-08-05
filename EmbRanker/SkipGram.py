import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

class SkipGram(nn.Module):
    def __init__(self,embed_manager,use_bert=False):
        super(SkipGram,self).__init__()
        self.embed_manager = embed_manager
        self.use_bert = use_bert

    def forward(self,pos_pair,neg_pair,pos_lens,neg_lens,pos_qd_mask,neg_qd_mask):
        #pos_pair, neg_pair, pos_qd_mask, neg_qd_mask = map(Tensor2Varible,[pos_pair,neg_pair,pos_qd_mask,neg_qd_mask])

        pos_score = self.pair_forward(pos_pair,pos_lens,pos_qd_mask)
        neg_score = self.pair_forward(neg_pair,neg_lens, neg_qd_mask)

        #print 'score:',pos_score.sum().item(),neg_score.sum().item()

        log_sigmoid_pos = F.logsigmoid(pos_score)
        log_sigmoid_neg = F.logsigmoid(-1 * neg_score)

        #print 'log score:', log_sigmoid_pos.sum().item(), log_sigmoid_neg.sum().item()
        loss = -1 * (log_sigmoid_pos + log_sigmoid_neg).sum()

        return loss


    def pair_forward(self,pair_data,pair_lens,qd_mask):
        '''
        :param pair_data: size: 2 * (bs,seq_len)
        :param qd_mask: (bs,2) 1 for query, 0 for doc
        :return:
        '''
        left_data,left_len = pair_data[0],pair_lens[0]
        right_data,right_len = pair_data[1],pair_lens[1]

        if self.use_bert:
            left_output_emb = self.embed_manager(left_data)
            right_output_emb = self.embed_manager(right_data)
        else:
            left_output_emb = self.embed_manager(left_data,left_len,qd_mask[:,0]) #(bs,embsize)
            right_output_emb = self.embed_manager(right_data,right_len, qd_mask[:, 1])

        score = torch.bmm(left_output_emb.unsqueeze(1),right_output_emb.unsqueeze(2)) #(bs,1,embsize) * (bs,embsize,1)

        #(bs,1)
        return score





