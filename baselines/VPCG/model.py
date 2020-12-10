import cPickle,os,tqdm
import networkx as nx
from utils import *
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics.rank_evaluations import *

PSCM_THRESHOLD = 0.5


class VPCG():
    def __init__(self,config):
        self.config = config
        self.word2id, self.id2word = cPickle.load(open(self.config['vocab_dict_file']))
        self.vocab_size = len(self.word2id)
        print 'Vocab_size: %d' % self.vocab_size

        self.graph_addr = self.config['graph_addr']
        self.qd_embed_addr = self.config['qd_embed_addr']
        self.word_embed_addr = self.config['word_embed_addr']
        self.topK = self.config['topK']

        self.evaluator = rank_eval()


    def build_graph(self):
        if os.path.exists(self.graph_addr):
            print 'Loading Graph Model...'
            self.G = cPickle.load(open(self.graph_addr))
            return

        print 'building graph...'
        train_addr = self.config['train_addr']
        trainFiles = os.listdir(train_addr)

        G = nx.Graph()

        for trainFile in tqdm.tqdm(trainFiles[:]):
            for i,line in enumerate(open(os.path.join(train_addr,trainFile))):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid,docid,query,title = elements[:4]
                pscm_label = float(elements[5])
                if pscm_label < PSCM_THRESHOLD:
                    continue

                G.add_edge(qid,docid,weight=pscm_label)
                query_term_ids = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))
                title_term_ids = map(lambda w: find_id(self.word2id, w), filter_title(title.split()))

                query_edges = map(lambda wid:(qid,'w'+str(wid)),query_term_ids)
                title_edges = map(lambda wid: (docid, 'w' + str(wid)), title_term_ids)

                G.add_edges_from(query_edges)
                G.add_edges_from(title_edges)

        print 'Saving New Graph Model...'
        cPickle.dump(G, open(self.graph_addr,'w'))

        self.G = G
        return

    def get_nodes(self,node_type):
        nodes = [node for node in list(self.G.nodes) if node[0] == node_type]
        return nodes

    def get_neighbors(self,node,node_type):
        neighbors = [neighbor for neighbor in self.G.neighbors(node) if neighbor[0] == node_type]
        return neighbors

    def get_weights(self,node,neighbors):
        weights = map(lambda t:self.G.get_edge_data(node,t)['weight'],neighbors)
        return weights

    def count_nodes(self):
        cq,cd,cw = 0,0,0 #count of query,doc,word
        for node in list(self.G.nodes):
            if node[0] == 'q':
                cq += 1
            elif node[1] == 'd':
                cd += 1
            elif node[2] == 'w':
                cw += 1

        self.cq,self.cd,self.cw = cq,cd,cw
        print 'Number of query: ',cq
        print 'Number of doc: ', cd
        print 'Number of word: ', cw

    def init_embed(self,node_type):
        embed = {}
        if node_type == 'w':
            for i in xrange(self.vocab_size):
                embed[i] = {i:1}
            return embed

        #for query or doc embedding
        nodes = self.get_nodes(node_type = node_type)
        for node in nodes:
            idx = int(node[1:])
            word_neighbors = self.get_neighbors(node,node_type='w')
            wids = map(lambda t:int(t[1:]), word_neighbors)
            embed[idx] = dict.fromkeys(wids,1)
        return embed

    def update(self,neigh_embeding,neighbors,weights):
        new_embed = defaultdict(lambda :0.)
        for neigh, weight in zip(neighbors, weights):
            neiId = int(neigh[1:])
            neigh_embed = neigh_embeding[neiId]
            for wid, feat in neigh_embed.iteritems():
                new_embed[wid] += weight * feat
        top_pairs = sorted(new_embed.items(),key=lambda t:-t[1])#sort by weight * feat
        normalize = sum(map(lambda t:t[1],top_pairs[:self.topK]))

        embed = {}
        for wid,feat in top_pairs[:self.topK]:
            embed[wid] = feat / (normalize + 1e-7)
        return embed


    def train(self):
        if os.path.exists(self.qd_embed_addr):
            print 'Loading qd embed...'
            self.query_embed,self.doc_embed = cPickle.load(open(self.qd_embed_addr))
            #print 'query_embed: ',self.query_embed.items()[:10]
            #print 'doc_embed: ',self.doc_embed.items()[:10]
            #exit(-1)
            return

        print 'training qd embedding...'
        self.query_embed = self.init_embed(node_type='q')#{w1:feat,w2:feat}
        self.doc_embed = self.init_embed(node_type='d')

        q_nodes = self.get_nodes(node_type='q')
        d_nodes = self.get_nodes(node_type='d')
        for iter in tqdm.tqdm(range(self.config['iteration'])):
            for node in tqdm.tqdm(q_nodes):
                qid = int(node[1:])
                neighbors = self.get_neighbors(node,node_type='d')
                weights = self.get_weights(node, neighbors)
                new_embed = self.update(self.doc_embed,neighbors,weights)
                self.query_embed[qid] = new_embed
            for node in tqdm.tqdm(d_nodes):
                did = int(node[1:])
                neighbors = self.get_neighbors(node,node_type='q')
                weights = self.get_weights(node, neighbors)
                new_embed = self.update(self.query_embed,neighbors,weights)
                self.doc_embed[did] = new_embed

        cPickle.dump((self.query_embed,self.doc_embed),open(self.qd_embed_addr,'w'))

    def train_word_embed(self):
        if os.path.exists(self.word_embed_addr):
            print 'Loading word embed...'
            self.word_embed = cPickle.load(open(self.word_embed_addr))
            return

        print 'training word embedding...'
        word_nodes = self.get_nodes(node_type='w')
        self.word_embed = self.init_embed(node_type='w')#{w:1}
        for node in tqdm.tqdm(word_nodes):
            wid = int(node[1:])
            q_nodes = self.get_neighbors(node,node_type='q')
            d_weights = defaultdict(lambda: 0.)
            for q_node in q_nodes:
                d_nodes = self.get_neighbors(q_node,node_type='d')
                tmp_weights = self.get_weights(q_node,d_nodes)
                for d_node,weight in zip(d_nodes,tmp_weights):
                    #did = int(d_node[1:])
                    d_weights[d_node] += weight
            neighbors,weights = d_weights.keys(),d_weights.values()
            new_word_embed = self.update(self.doc_embed,neighbors,weights)
            self.word_embed[wid] = new_word_embed

        cPickle.dump(self.word_embed,open(self.word_embed_addr,'w'))

    def get_embed(self,node,words):
        node_id = int(node[1:])
        node_type = node[0]
        if node_type == 'q' and node_id in self.query_embed:
            #print '-------------------------------------run here q'
            return self.query_embed[node_id]
        elif node_type == 'd' and node_id in self.doc_embed:
            #print '-------------------------------------run here d '
            return self.doc_embed[node_id]
        else:
            #print '-------------------------------------run here w'
            word_idx = map(lambda w: find_id(self.word2id, w), filter_title(words.split()))
            word_idx = map(lambda w:'w'+str(w),word_idx)
            return self.construct_embed(word_idx)

    def construct_embed(self,word_idx):
        weights = [1 for i in range(len(word_idx))]
        return self.update(self.word_embed,word_idx,weights)

    def predict(self,embed_q,embed_d):
        numerator = 0.
        for wid,feat in embed_q.iteritems():
            if wid in embed_d:
                numerator += embed_d[wid]
        try:
            norm_a = np.sqrt(np.sum(map(lambda t:t**2,embed_q.values())))
            norm_b = np.sqrt(np.sum(map(lambda t:t**2,embed_d.values())))
        except:
            print 'embed_q:', embed_q
            print 'embed_d:', embed_d

        return numerator / (norm_a * norm_b + 1e-7)

    def test(self,label_type='PSCM',save_result=False):
        model_id = model2id(label_type)
        data_addr = self.config['test_addr']
        text_list = os.listdir(data_addr)
        results = defaultdict(list)

        print 'testing...'
        for text_id in tqdm.tqdm(text_list[:]):
            preds = [];gts = [];
            for i, line in enumerate(open(os.path.join(data_addr, text_id))):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid, docid, query, title = elements[:4]
                labels = map(float, elements[-7:])
                label = labels[model_id]

                q_embed = self.get_embed(qid, query)
                d_embed = self.get_embed(docid, title)

                rel = self.predict(q_embed,d_embed)

                preds.append(rel)
                gts.append(labels[model_id])

            gts = map(lambda t: score2cutoff(label_type, t), gts)
            result = self.evaluator.eval(gts, preds)
            for k, v in result.items():
                results[k].append(v)

        performances = {}
        for k, v in results.items():
            performances[k] = np.mean(v)

        print '-----------------------------Performance:-----------------------------'
        print 'Label: ', label_type
        print performances

        if save_result:
            path = './results/' + label_type + '_result.pkl'
            cPickle.dump(results, open(path, 'w'))

        return performances








    '''
    def train_word_weights(self):
        q_nodes = self.get_nodes(node_type='q')
        d_nodes = self.get_nodes(node_type='d')

        self.weight_word = np.random.random([self.vocab_size])
        for iter in tqdm.tqdm(range(self.config['iteration'])):
            for node in q_nodes:
                qid = int(node[1:])
                target_embed = self.query_embed[qid]
                neighbor_words = self.get_neighbors(node, node_type='w')

    def objective(self,target_embed,neighbor_words):
        U_list = []
        for unit in neighbor_words:
            unit_id = int(unit[1:])
            w_embed = self.word_embed[unit_id]  # {wid:feat,...}

            wids,feats = w_embed.keys(),w_embed.values()
            index = torch.LongTensor([wids,[0 for i in range(wids)]])
            value = torch.FloatTensor(feats)
            U = Variable(torch.sparse.FloatTensor(index,value,torch.Size([self.vocab_size,1])),requires_grad=False)

            W = self.weight_word(Variable(torch.LongTensor([unit_id]))).view(-1,1)#(1,1)

            WU = U.mm(W)
            U_list.append(WU)
        pred =
    '''





































