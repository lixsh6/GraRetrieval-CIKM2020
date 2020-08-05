import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import cPickle
from transformer import *
from pytorch_pretrained_bert import BertModel

class Embed(nn.Module):
    def __init__(self,vocab_size, config):
        super(Embed, self).__init__()

        self.embsize = config['embsize']
        self.text_dim = config['text_dim']
        self.term_embedding = nn.Embedding(vocab_size, self.embsize)#padding_idx=0

        pre_word_embeds_addr = config['emb'] if 'emb' in config else None
        if pre_word_embeds_addr != None and 'test_mode' not in config:
            print 'Loading word embeddings'
            pre_word_embeds = cPickle.load(open(pre_word_embeds_addr))
            print 'pre_word_embeds size: ', pre_word_embeds.shape
            self.term_embedding.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))

        max_seq_len = max(config['max_q_len'],config['max_d_len'])
        self.ranker = config['ranker']
        if self.ranker == 'ARCI':
            self.query_encoder = KimCNN(64, 3, 50, self.text_dim)
            self.doc_encoder = KimCNN(64, 3, 50, self.text_dim)
        elif self.ranker == 'DSSM':
            self.query_encoder = MLP(self.embsize,self.text_dim)
            self.doc_encoder = MLP(self.embsize, self.text_dim)
        elif self.ranker == 'LSTM-RNN':
            self.query_encoder = LSTM(self.embsize,self.text_dim)
            self.doc_encoder = LSTM(self.embsize, self.text_dim)
        elif self.ranker == 'TRANSFORMER':
            self.query_encoder = Encoder(max_seq_len, self.term_embedding, num_layers=2,\
                                     model_dim=self.embsize, num_heads=5, ffn_dim=64,dropout=0)
            self.doc_encoder = Encoder(max_seq_len, self.term_embedding, num_layers=2,\
                                     model_dim=self.embsize, num_heads=5, ffn_dim=64,dropout=0)
        elif self.ranker == 'BERT':
            self.bert_model = BertModel.from_pretrained(config['BERT_folder'])
        else:
            raise Exception('error model name')


    def forward(self, data, lens=None, q_mask=None):
        '''
        :param data: (bs,max_seq_len) one side of (q,d) pair
        :param mask: 1 for query, 0 for doc (bs,)
        :return:
        '''
        data = Tensor2Varible(torch.LongTensor(data))

        if self.ranker == 'BERT':
            segments_ids = Tensor2Varible(torch.zero_likes(data))
            output = self.embed_manager.bert_model(data, segments_ids)
        else:
            lens = Tensor2Varible(torch.LongTensor(lens))
            q_mask = Tensor2Varible(torch.FloatTensor(q_mask))

            #print 'data: ',data.size()
            if self.ranker == 'LSTM-RNN':
                data_emb = self.term_embedding(data)
                query_embed = self.query_encoder(data_emb,lens)
                doc_embed = self.doc_encoder(data_emb,lens)
            elif self.ranker == 'TRANSFORMER':
                query_embed = self.query_encoder(data, lens)
                doc_embed = self.doc_encoder(data, lens)
            else:
                data_emb = self.term_embedding(data)
                query_embed = self.query_encoder(data_emb)
                doc_embed = self.doc_encoder(data_emb)

            #print 'query_embed:',query_embed.size()
            #print 'doc_embed: ',doc_embed.size()
            q_mask = q_mask.view(-1, 1)
            output = query_embed * q_mask + doc_embed * (1.0 - q_mask)

        return output











