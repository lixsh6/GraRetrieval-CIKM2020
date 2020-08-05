# encoding=utf8
import numpy as np
import cPickle,re
import random,time
import tqdm,os
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer

import sys
reload(sys)
sys.setdefaultencoding('utf8')


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


#FOR BERT
def word_split(text):
    text = unicode(text, 'utf-8')
    return [i.strip() for i in text if len(i.strip()) > 0]

def bert_convert_ids(text_input,tokenizer):
    text_list = ['[CLS]'] + word_split(text_input) + ['[SEP]']
    text = ' '.join(text_list).decode('utf8')
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens

def model2id(model_name):
    #print 'model_name: ',model_name
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

class DataGenerator():
    def __init__(self, config, message):
        #super(DataGenerator, self).__init__(config)
        print 'Data Generator initializing...'
        self.config = config

        self.word2id, self.id2word = cPickle.load(open(config['vocab_dict_file']))
        #self.node2id, self.id2node, self.id2wordid = cPickle.load(open(config['node_dict_file']))

        self.vocab_size = len(self.word2id)
        print 'Vocab_size: %d' % self.vocab_size

        self.train_rank_addr = config['train_rank_addr']
        self.test_mode = config['test_mode']
        self.message = message
        self.use_bert = False

        if config['ranker'] == 'BERT':
            self.use_bert = True
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config['BERT_folder'], config['BERT_VOCAB']))




    def ranking_pair_reader(self,batch_size):

        click_model = self.config['click_model']
        model_id = model2id(click_model)
        text_list = os.listdir(self.train_rank_addr)

        query_batch, doc_pos_batch, doc_neg_batch, doc_pos_length, doc_neg_length = [],[],[],[],[]
        query_ids, doc_pos_ids, doc_neg_ids = [], [], []

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len,max_q_len)

        print 'text_list Size: ',len(text_list)
        epoch = 0
        last_time = time.time()

        pad_seq_ = self.pad_seq_bert if self.use_bert else self.pad_seq

        while True:
            random.shuffle(text_list)
            now = time.time()
            print ('----------Epoch: %d\t Time cost:%.4f----------' % (epoch,now - last_time))
            last_time = now
            epoch += 1

            for text_id in text_list:
                documents = [];relevances = [];lengths = [];doc_ids = []

                for i,line in enumerate(open(os.path.join(self.train_rank_addr,text_id))):
                    if i == 0:
                        continue
                    elements = line.strip().split('\t')
                    qid = elements[0]
                    did = elements[1]
                    query = elements[2]

                    if self.use_bert:
                        title_words = elements[3]
                        title_idx = bert_convert_ids(title_words,self.tokenizer)
                        if len(title_idx) <= 2:
                            continue
                    else:
                        title_words = filter_title(elements[3].split())
                        title_idx = map(lambda w: find_id(self.word2id, w), title_words)[:max_seq_len]
                        if len(title_idx) == 0:
                            continue


                    documents.append(title_idx)
                    doc_ids.append(did)

                    labels = map(float,elements[-6:])
                    relevances.append(labels[model_id])
                    lengths.append(len(title_idx))

                if self.use_bert:
                    query_idx = bert_convert_ids(query, self.tokenizer)
                else:
                    query_idx = map(lambda w: find_id(self.word2id, w),filter_title(query.split()))[:max_seq_len]
                #cur_q_neibs = self.get_neibs(qid)

                for i in range(len(documents) - 1):
                    for j in range(i + 1, len(documents)):
                        pos_i, neg_i = i, j
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
                        doc_pos_batch.append(pos_doc)
                        doc_pos_length.append(pos_len)

                        doc_neg_batch.append(neg_doc)
                        doc_neg_length.append(neg_len)

                        query_ids.append(qid)
                        doc_pos_ids.append(doc_ids[pos_i])
                        doc_neg_ids.append(doc_ids[neg_i])

                        if len(query_batch) >= batch_size:
                            query_lengths = np.array([len(s) for s in query_batch])
                            max_query_len = max(max(query_lengths),3)
                            max_pos_doc_len = max(max(doc_pos_length),3)
                            max_neg_doc_len = max(max(doc_neg_length),3)

                            doc_pos_length = np.array(doc_pos_length)
                            doc_neg_length = np.array(doc_neg_length)

                            query_batch = np.array([pad_seq_(s, max_query_len) for s in query_batch])
                            doc_pos_batch = np.array([pad_seq_(d[:max_pos_doc_len], max_pos_doc_len) for d in doc_pos_batch])
                            doc_neg_batch = np.array([pad_seq_(d[:max_neg_doc_len], max_neg_doc_len) for d in doc_neg_batch])

                            yield (query_batch, doc_pos_batch, doc_neg_batch, query_lengths, doc_pos_length, doc_neg_length, \
                                   query_ids,doc_pos_ids,doc_neg_ids)

                            query_batch, doc_pos_batch, doc_neg_batch, doc_pos_length, doc_neg_length = [], [], [], [], []
                            query_ids, doc_pos_ids, doc_neg_ids = [], [], []

    def ranking_point_reader(self,data_addr,is_test=False,label_type='PSCM'):
        model_id = model2id(label_type)
        text_list = os.listdir(data_addr)

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len, max_q_len)

        if self.config['test_mode']:
            text_list = text_list[:]

        pad_seq_ = self.pad_seq_bert if self.use_bert else self.pad_seq

        for text_id in tqdm.tqdm(text_list):
            doc_batch, gt_rels, doc_lengths,query_ids, doc_ids = [], [], [], [], []

            for i,line in enumerate(open(os.path.join(data_addr, text_id))):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid = elements[0]
                did = elements[1]
                query = elements[2]

                if self.use_bert:
                    title_words = elements[3]
                    title_idx = bert_convert_ids(title_words, self.tokenizer)
                    if len(title_idx) <= 2:
                        continue
                else:
                    title_words = filter_title(elements[3].split())
                    title_idx = map(lambda w: find_id(self.word2id, w), title_words)[:max_seq_len]
                    if len(title_idx) == 0:
                        continue

                #title_idx = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_seq_len]
                #if len(title_idx) == 0:
                #    continue

                index = -7 if is_test else -6
                labels = map(float,elements[index:])
                doc_batch.append(title_idx)
                doc_lengths.append(len(title_idx))
                query_ids.append(qid)
                doc_ids.append(did)

                gt_rels.append(labels[model_id])

            if self.use_bert:
                query_idx = bert_convert_ids(query, self.tokenizer)
            else:
                query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))[:max_seq_len]

            query_idx = pad_seq_(query_idx, max(max_q_len, 3))

            item_size = len(doc_batch)
            query_batch = np.array([query_idx for i in range(item_size)])
            query_lengths = np.array([len(s) for s in query_batch])

            doc_lengths = np.array(doc_lengths)
            max_doc_len = max(max(doc_lengths),3)
            doc_batch = np.array([pad_seq_(d[:max_doc_len], max_doc_len) for d in doc_batch])

            #print 'query_batch:',query_batch.shape,doc_batch.shape
            yield (query_ids,doc_ids, query_batch, doc_batch, query_lengths, doc_lengths, gt_rels)



    def pad_seq(self, seq, max_length, PAD_token=0):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    def pad_seq_bert(self, seq, max_length, PAD_token=103):
        # id of [MASK] is 103
        if len(seq) > max_length:
            seq = seq[:max_length - 1] + [seq[-1]]
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq