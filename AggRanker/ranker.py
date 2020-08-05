import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *



class Ranker(nn.Module):
    def __init__(self,config,embed_manager):
        super(Ranker, self).__init__()
        self.embed_manager = embed_manager

        self.embsize = config['embsize']
        self.text_dim = config['text_dim']
        self.ranker = config['ranker']
        self.aggregation_type = config['aggregation_type']

        if self.aggregation_type == 'GCN':
            self.linear = nn.Linear(self.text_dim, self.text_dim)
            self.lin_out = nn.Linear(self.text_dim * 2, 1)
        elif self.aggregation_type == 'GraphSage':
            self.linear1 = nn.Linear(self.text_dim * 2, self.text_dim)
            self.linear2 = nn.Linear(self.text_dim * 2, self.text_dim)
            self.drop1 = nn.Dropout(0.1)
            self.drop2 = nn.Dropout(0.1)
            self.drop_q = nn.Dropout(0.2)
            self.drop_d = nn.Dropout(0.2)
            self.lin_out = nn.Linear(self.text_dim * 4, 1)
        else:
            self.linear1 = nn.Linear(self.text_dim, self.text_dim)
            self.linear2 = nn.Linear(self.text_dim, self.text_dim)


        if self.ranker != 'BERT':
            self.term_embedding = embed_manager.term_embedding
        else:
            self.output_layer_index = config['output_layer_index']
        self.activation = nn.Tanh()#nn.LeakyReLU(0.2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward(self,qids, dids, query_batch,doc_batch, q_lens=None, d_lens=None):
        query_var = Tensor2Varible(torch.LongTensor(query_batch))
        doc_var = Tensor2Varible(torch.LongTensor(doc_batch))

        if self.ranker == 'BERT':
            query_segments_ids = Tensor2Varible(torch.zeros(query_var.size(), dtype=torch.long))
            doc_segments_ids = Tensor2Varible(torch.zeros(doc_var.size(), dtype=torch.long))
            query_encoded_layers, _ = self.embed_manager.bert_model(query_var,query_segments_ids)
            doc_encoded_layers, _ = self.embed_manager.bert_model(doc_var,doc_segments_ids)
            query_embed = query_encoded_layers[self.output_layer_index][:,0,:]
            doc_embed = doc_encoded_layers[self.output_layer_index][:,0,:]
        else:
            q_lens = Tensor2Varible(torch.LongTensor(q_lens))
            d_lens = Tensor2Varible(torch.LongTensor(d_lens))


        #Only use for non bert
        if self.ranker == 'LSTM-RNN':
            query_var = self.term_embedding(query_var)
            doc_var = self.term_embedding(doc_var)
            query_embed = self.embed_manager.query_encoder(query_var,q_lens)  # (bs,odim)
            doc_embed = self.embed_manager.doc_encoder(doc_var,d_lens)
        elif self.ranker == 'ARCI' or self.ranker == 'DSSM':
            query_var = self.term_embedding(query_var)
            doc_var = self.term_embedding(doc_var)
            query_embed = self.embed_manager.query_encoder(query_var)  # (bs,odim)
            doc_embed = self.embed_manager.doc_encoder(doc_var)
        elif self.ranker == 'TRANSFORMER':
            query_embed = self.embed_manager.query_encoder(query_var, q_lens)
            doc_embed = self.embed_manager.doc_encoder(doc_var, d_lens)

        #query_neib_embed = self.embed_manager(qids)
        #doc_neib_embed = self.embed_manager(dids)
        query_neib_embed = self.drop1(self.embed_manager(qids))#self.drop1()
        doc_neib_embed = self.drop2(self.embed_manager(dids))#self.drop2()

        if self.aggregation_type == 'GCN':
            #query_out = self.activation(self.linear(query_embed)) # + query_neib_embed
            #doc_out = self.activation(self.linear(doc_embed)) # + doc_neib_embed
            query_out = query_embed + query_neib_embed
            doc_out = doc_embed + doc_neib_embed
        elif self.aggregation_type == 'GraphSage':
            #query_out = self.activation(self.linear1(torch.cat([query_embed, query_neib_embed],dim=1)))
            #doc_out = self.activation(self.linear2(torch.cat([doc_embed, doc_neib_embed],dim=1)))
            #print 'size:',query_embed.size()
            query_out = self.drop_q(torch.cat([query_embed, query_neib_embed],dim=1))#
            doc_out = self.drop_d(torch.cat([doc_embed, doc_neib_embed],dim=1))#
        else:
            query_out = self.activation(self.linear1(query_embed + query_neib_embed)) + \
                self.activation(self.linear2(query_embed.mul(query_neib_embed)))
            doc_out = self.activation(self.linear1(doc_embed + doc_neib_embed)) + \
                        self.activation(self.linear2(doc_embed.mul(doc_neib_embed)))

        dm_score = self.lin_out(torch.cat([query_out, doc_out], dim=-1))
        #dm_score = self.cos(query_out,doc_out)
        return dm_score








