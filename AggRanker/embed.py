import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import *
import cPickle
from transformer import *

#Based on a fixed neighborhood dict

class Embed(nn.Module):
    def __init__(self,vocab_size, config):
        super(Embed, self).__init__()

        self.embsize = config['embsize']
        self.ranker = config['ranker']
        self.neighbor_dict = cPickle.load(open(config['neib_dict_addr']))
        self.text_dim = config['text_dim']
        max_seq_len = max(config['max_q_len'], config['max_d_len'])

        if self.ranker != 'BERT':

            self.term_embedding = nn.Embedding(vocab_size, self.embsize)#padding_idx=0
            pre_word_embeds_addr = config['emb'] if 'emb' in config else None
            if pre_word_embeds_addr != None and 'test_mode' not in config:
                print 'Loading word embeddings'
                pre_word_embeds = cPickle.load(open(pre_word_embeds_addr))
                print 'pre_word_embeds size: ', pre_word_embeds.shape
                self.term_embedding.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))

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
                self.query_encoder = Encoder(max_seq_len, self.term_embedding, num_layers=2, \
                                             model_dim=self.embsize, num_heads=5, ffn_dim=64,dropout=0)
                self.doc_encoder = Encoder(max_seq_len, self.term_embedding, num_layers=2, \
                                           model_dim=self.embsize, num_heads=5, ffn_dim=64,dropout=0)
            self.activation = nn.Tanh()
            self.content_encoder = ContentCNN(self.term_embedding, self.query_encoder, self.doc_encoder, config)

        else:
            self.bert_model = BertModel.from_pretrained(config['BERT_folder'])
            self.content_encoder = BERT_encoder(self.bert_model,config)


        neig_sizes = [4,8,10,12,14]

        self.encoder1 = Aggregator(self.content_encoder, self.neighbor_dict, self.text_dim, self.text_dim,neig_sizes[1]) #8
        self.encoder2 = Aggregator(lambda nodes:self.encoder1(nodes),self.neighbor_dict, self.text_dim, self.text_dim,neig_sizes[0]) #4
        #self.encoder3 = Aggregator(lambda nodes: self.encoder2(nodes), self.neighbor_dict, self.text_dim,self.text_dim,neig_sizes[1])  # 4
        #self.encoder4 = Aggregator(lambda nodes: self.encoder3(nodes), self.neighbor_dict, self.text_dim,self.text_dim,neig_sizes[0])
        #self.encoder5 = Aggregator(lambda nodes: self.encoder4(nodes), self.neighbor_dict, self.text_dim,self.text_dim,neig_sizes[0])
        #self.encoder[n] = Aggregator(lambda nodes:self.encoder[n-1](nodes),self.neighbor_dict, self.text_dim, self.text_dim)
        self.encoder = self.encoder2 #self.encoder4
    def forward(self, node_ids):
        '''
        :param data_ids:(bs)
        :return:
        '''

        output_embed = self.encoder(node_ids)

        return output_embed # node_num * emb_dim


class Aggregator(nn.Module):
    def __init__(self,features,neighbor_dict,input_dim,out_dim,sample_size=-1):
        super(Aggregator, self).__init__()
        self.features = features
        self.neighbor_dict = neighbor_dict

        self.linear = nn.Linear(input_dim, out_dim)
        self.activation = nn.Tanh()
        self.drop = nn.Dropout(0.9)
        self.sample_size = sample_size

    def sample_neighs(self,nodes,size):
        if size == -1 or len(nodes) <= size:
            return nodes
        else:
            nodes = list(np.random.choice(nodes, size, replace=False))
            return nodes

    def forward(self,nodes):
        to_neighs = [self.sample_neighs(self.neighbor_dict[node],self.sample_size) \
                     if node in self.neighbor_dict else ['0'] for node in nodes]
        samp_neighs = [samp_neigh + [nodes[i]] for i, samp_neigh in enumerate(to_neighs)] # + [nodes[i]]

        unique_nodes_list = exclusive_combine(samp_neighs)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        node_qd_mask = [1 if nid[0] == 'q' else 0 for nid in unique_nodes_list]#q:1, d:0
        # The mask for aggregation
        mask = Tensor2Varible(torch.zeros(len(samp_neighs), len(unique_nodes), requires_grad=False))
        # The connections
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # Normalize
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        embed_matrix = self.features(unique_nodes_list)

        neigh_feats = mask.mm(embed_matrix)

        output_embed = self.activation(self.drop(self.linear(neigh_feats)))
        return output_embed  # node_num * text_dim

class BERT_encoder(nn.Module):
    def __init__(self,bert_model,config):
        super(BERT_encoder, self).__init__()
        self.max_seq_len = max(config['max_q_len'], config['max_d_len'])
        self.id2textid_dict = cPickle.load(open(config['bert_id2textid_dict']))
        self.bert_model = bert_model
        self.output_layer_index = config['output_layer_index']

    def forward(self,nodes_list):
        node_texts = []
        for nid in nodes_list:
            if nid in self.id2textid_dict:
                node_texts.append(pad_seq_bert(self.id2textid_dict[nid],self.max_seq_len))
            else:
                #id of unk is 100
                node_texts.append(pad_seq_bert([100], self.max_seq_len))

        segments_ids = [0] * self.max_seq_len
        tokens_tensor = Tensor2Varible(torch.tensor(node_texts))
        segments_tensors = Tensor2Varible(torch.tensor([segments_ids] * len(node_texts)))

        encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)
        ##[batch,sequence,embedding]
        #output: [batch,embedding]
        return encoded_layers[self.output_layer_index][:,0,:]

class ContentCNN(nn.Module):
    def __init__(self,term_embedding,query_encoder,doc_encoder, config):
        super(ContentCNN, self).__init__()
        self.term_embedding = term_embedding
        self.qid2text_dict = cPickle.load(open(config['query_dict_addr']))
        self.did2text_dict = cPickle.load(open(config['doc_dict_addr']))
        self.max_seq_len = max(config['max_q_len'],config['max_d_len'])

        self.ranker = config['ranker']
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def get_max_len(self,nodes_list):
        max_len = 0
        for nid in nodes_list:
            if nid in self.qid2text_dict:
                max_len = max(max_len,len(self.qid2text_dict[nid][:self.max_seq_len]))
            elif nid in self.did2text_dict:
                max_len = max(max_len, len(self.did2text_dict[nid][:self.max_seq_len]))
        return max_len

    def forward(self,nodes_list):
        node_texts = [];lens = [];q_mask = []
        max_len = max(self.get_max_len(nodes_list),3)

        for nid in nodes_list:
            if nid in self.qid2text_dict:
                text_ids = self.qid2text_dict[nid][:self.max_seq_len]
                node_texts.append(pad_seq(text_ids,max_len))
                lens.append(len(text_ids))
                q_mask.append(1)
            elif nid in self.did2text_dict:
                text_ids = self.did2text_dict[nid][:self.max_seq_len]
                node_texts.append(pad_seq(text_ids,max_len))
                lens.append(len(text_ids))
                q_mask.append(0)
            else:
                node_texts.append(pad_seq([1],max_len))
                lens.append(1)
                q_mask.append(2)

        #print 'node_texts: ',node_texts,
        #print 'lens: ',lens,max_len
        #print 'q_mask:',q_mask
        data = Tensor2Varible(torch.LongTensor(node_texts))
        lens = Tensor2Varible(torch.LongTensor(lens))
        q_mask = Tensor2Varible(torch.FloatTensor(q_mask))


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

        q_mask = q_mask.view(-1, 1)
        output = query_embed * q_mask + doc_embed * (1.0 - q_mask)

        return output #(bs,text_dim)



















