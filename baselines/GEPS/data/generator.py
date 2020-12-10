# encoding=utf8
import numpy as np
import cPickle,re
import random
import tqdm,os
from collections import defaultdict


useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・']

def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)
    return words

def find_id(word_dict,word):
    return word_dict[word] if word in word_dict else 1


def model2id(model_name):
    #print 'model_name: ',model_name
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

class DataGenerator():
    def __init__(self, config):
        #super(DataGenerator, self).__init__(config)
        print 'Data Generator initializing...'
        self.config = config

        self.word2id, self.id2word = cPickle.load(open(config['vocab_dict_file']))
        self.vocab_size = len(self.word2id)
        print 'Vocab_size: %d' % self.vocab_size

        self.q2id, self.id2q = cPickle.load(open(config['query_dict_file']))
        self.query_size = len(self.q2id)
        print 'Query_size: %d' % self.query_size

        self.d2id, self.id2d = cPickle.load(open(config['doc_dict_file']))
        self.doc_size = len(self.d2id)
        print 'Doc_size: %d' % self.doc_size

        self.train_rank_addr = config['train_rank_addr']


    def ranking_pair_reader(self,batch_size):

        click_model = self.config['click_model']
        model_id = model2id(click_model)
        text_list = os.listdir(self.train_rank_addr)

        query_batch, doc_pos_batch, doc_neg_batch, doc_pos_length, doc_neg_length = [],[],[],[],[]
        query_id_batch,docid_pos_batch,docid_neg_batch = [],[],[]

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len,max_q_len)

        while True:
            random.shuffle(text_list)
            for text_id in text_list:
                documents = [];relevances = [];lengths = [];docIds = []
                for i,line in enumerate(open(os.path.join(self.train_rank_addr,text_id))):
                    if i == 0:
                        continue
                    elements = line.strip().split('\t')
                    query = elements[2]
                    qid = elements[0]
                    docid = elements[1]

                    title_idx = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_seq_len]

                    if len(title_idx) == 0:
                        continue
                    documents.append(title_idx)

                    labels = map(float,elements[-6:])
                    relevances.append(labels[model_id])
                    lengths.append(len(title_idx))
                    docIds.append(self.d2id[docid] if docid in self.d2id else 0)

                query_idx = map(lambda w: find_id(self.word2id, w),filter_title(query.split()))[:max_seq_len]
                query_id = self.q2id[qid] if qid in self.q2id else 0

                for i in range(len(documents) - 1):
                    for j in range(i + 1, len(documents)):
                        pos_i,neg_i = i, j
                        y_diff = relevances[pos_i] - relevances[neg_i]
                        if abs(y_diff) < self.config['min_score_diff']:
                            continue
                        if y_diff < 0:
                            pos_i, neg_i = neg_i, pos_i

                        pos_doc = documents[pos_i]
                        neg_doc = documents[neg_i]

                        pos_len = lengths[pos_i]
                        neg_len = lengths[neg_i]

                        query_batch.append(query_idx)
                        query_id_batch.append(query_id)

                        doc_pos_batch.append(pos_doc)
                        doc_pos_length.append(pos_len)
                        docid_pos_batch.append(docIds[pos_i])


                        doc_neg_batch.append(neg_doc)
                        doc_neg_length.append(neg_len)
                        docid_neg_batch.append(docIds[neg_i])

                        if len(query_batch) >= batch_size:
                            query_lengths = np.array([len(s) for s in query_batch])
                            max_query_len = max(max(query_lengths),3)
                            max_pos_doc_len = max(max(doc_pos_length),3)
                            max_neg_doc_len = max(max(doc_neg_length),3)
                            #query_lengths = np.array(self.pad_seq(query_lengths,max_seq_len))
                            query_batch = np.array([self.pad_seq(s, max_query_len) for s in query_batch])
                            doc_pos_batch = np.array([self.pad_seq(d[:max_pos_doc_len], max_pos_doc_len) for d in doc_pos_batch])
                            doc_neg_batch = np.array([self.pad_seq(d[:max_neg_doc_len], max_neg_doc_len) for d in doc_neg_batch])

                            query_id_batch = np.array(query_id_batch)
                            docid_pos_batch = np.array(docid_pos_batch)
                            docid_neg_batch = np.array(docid_neg_batch)

                            yield (query_batch, doc_pos_batch, doc_neg_batch, query_id_batch, docid_pos_batch, docid_neg_batch)

                            query_batch, doc_pos_batch, doc_neg_batch, doc_pos_length, doc_neg_length = [], [], [], [], [];
                            query_id_batch, docid_pos_batch, docid_neg_batch = [], [], []

    def ranking_point_reader(self,data_addr,is_test=False,label_type='PSCM'):
        model_id = model2id(label_type)
        text_list = os.listdir(data_addr)

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len, max_q_len)

        if 'test_mode' in self.config:
            text_list = text_list[:30]

        for text_id in tqdm.tqdm(text_list[:]):
            doc_batch, gt_rels, doc_lengths, docid_batch = [], [], [], []
            for i,line in enumerate(open(os.path.join(data_addr, text_id))):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                query = elements[2]
                qid = elements[0]
                docid = elements[1]

                title_idx = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_seq_len]
                if len(title_idx) == 0:
                    continue

                index = -7 if is_test else -6
                labels = map(float,elements[index:])
                doc_batch.append(title_idx)
                doc_lengths.append(len(title_idx))
                gt_rels.append(labels[model_id])
                docid_batch.append(self.d2id[docid] if docid in self.d2id else 0)

            query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))[:max_seq_len]
            query_idx = self.pad_seq(query_idx, max_q_len)

            query_batch = [query_idx for i in range(len(doc_batch))]
            #query_lengths = np.array([len(s) for s in query_batch])

            doc_lengths = np.array(doc_lengths)
            max_doc_len = max(max(doc_lengths),3)
            doc_batch = np.array([self.pad_seq(d, max_doc_len) for d in doc_batch])
            docid_batch = np.array(docid_batch)

            yield (query_batch, doc_batch, docid_batch, gt_rels)




    def pad_seq(self, seq, max_length, PAD_token=0):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq