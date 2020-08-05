import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *


class Ranker(nn.Module):
    def __init__(self,config,embed_manager):
        super(Ranker, self).__init__()
        self.embed_manager = embed_manager
        self.term_embedding = embed_manager.term_embedding
        #self.batchNorm = nn.BatchNorm1d(1)
        #self.linear = nn.Linear(self.kernel_num,1)
        self.linear = nn.Linear(config['embsize'] * 2, 1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.ranker = config['ranker']
        self.drop_q = nn.Dropout(0.2)#0.8
        self.drop_d = nn.Dropout(0.2)

    def dm_match(self,query_var,q_lens_var,doc_var,d_lens_var):
        '''
        :param query_var:
        :param q_lens_var:
        :param doc_var:
        :param d_lens_var:
        :return: destributed score
        '''

        if self.ranker == 'BERT':
            query_segments_ids = Tensor2Varible(torch.zero_likes(query_var))
            doc_segments_ids = Tensor2Varible(torch.zero_likes(doc_var))
            query_encoded_layers, _ = self.embed_manager.bert_model(query_var, query_segments_ids)
            doc_encoded_layers, _ = self.embed_manager.bert_model(doc_var, doc_segments_ids)
            query_embed = query_encoded_layers[self.output_layer_index][:, 0, :]
            doc_embed = doc_encoded_layers[self.output_layer_index][:, 0, :]

        elif self.ranker == 'LSTM-RNN':
            query_embed = self.embed_manager.query_encoder(self.term_embedding(query_var),q_lens_var)
            doc_embed = self.embed_manager.doc_encoder(self.term_embedding(doc_var),d_lens_var)
        elif self.ranker == 'TRANSFORMER':
            query_embed = self.embed_manager.query_encoder(query_var, q_lens_var)
            doc_embed = self.embed_manager.doc_encoder(doc_var, d_lens_var)
            query_embed = self.drop_q(query_embed)
            doc_embed = self.drop_d(doc_embed)
        else:
            query_embed = self.embed_manager.query_encoder(self.term_embedding(query_var))
            doc_embed = self.embed_manager.doc_encoder(self.term_embedding(doc_var))

            query_embed = self.drop_q(query_embed)
            doc_embed = self.drop_d(doc_embed)
        #dm_score =  self.cos(query_embed,doc_embed)

        dm_score = self.linear(torch.cat([query_embed,doc_embed], dim=-1))
        return dm_score


    def get_mask(self, input_q, input_d):
        query_mask = 1 - torch.eq(input_q, 0).float()
        sent_mask = 1 - torch.eq(input_d, 0).float()
        input_mask = torch.bmm(query_mask.unsqueeze(-1), sent_mask.unsqueeze(1))
        return input_mask

    def forward(self,query_batch,query_lengths,doc_batch,doc_lengths):
        query_var = Tensor2Varible(torch.LongTensor(query_batch))
        q_lens_var = Tensor2Varible(torch.LongTensor(query_lengths))

        doc_var = Tensor2Varible(torch.LongTensor(doc_batch))
        d_lens_var = Tensor2Varible(torch.LongTensor(doc_lengths))


        dm_score = self.dm_match(query_var, q_lens_var, doc_var, d_lens_var)

        score = dm_score

        #(bs,1)
        return score







