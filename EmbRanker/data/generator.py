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


def model2id(model_name):
    #print 'model_name: ',model_name
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

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

class DataGenerator():
    def __init__(self, config, message):
        #super(DataGenerator, self).__init__(config)
        print 'Data Generator initializing...'
        self.config = config

        self.word2id, self.id2word = cPickle.load(open(config['vocab_dict_file']))
        self.vocab_size = len(self.word2id)
        print 'Vocab_size: %d' % self.vocab_size

        self.pair_addr = config['train_skg_pair_addr']
        self.train_rank_addr = config['train_rank_addr']
        self.message = message
        self.use_bert = False

        if config['ranker'] == 'BERT':
            self.use_bert = True
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config['BERT_folder'], config['BERT_VOCAB']))
        self.neg_skg_pairs, self.neg_skg_masks = self.load_neg_SkG_pairs(config['neg_skg_pairs_addr'])

    def load_neg_SkG_pairs(self,addr):
        print 'loading negative sk_gram pairs...'
        neg_pairs = [[],[]];qd_mask = []

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']
        max_seq_len = max(max_q_len, max_d_len)

        for line in open(addr):
            elements = line.strip().split('\t')
            uid,vid,utext,vtext = elements

            if self.use_bert:
                u_idx = bert_convert_ids(utext,self.tokenizer)
                v_idx = bert_convert_ids(vtext, self.tokenizer)
                if len(u_idx) <= 2 or len(v_idx) <= 2:
                    continue
            else:
                u_idx = map(lambda w:find_id(self.word2id,w), filter_title(utext.split()))[:max_seq_len]
                v_idx = map(lambda w: find_id(self.word2id, w), filter_title(vtext.split()))[:max_seq_len]

                if len(u_idx) == 0 or len(v_idx) == 0:
                    continue

            u_mask = 0 if uid[0] == 'd' else 1
            v_mask = 0 if vid[0] == 'd' else 1

            qd_mask.append([u_mask,v_mask])

            neg_pairs[0].append(u_idx)
            neg_pairs[1].append(v_idx)

        print 'Negative skg pairs loaded'
        print 'Neg Pairs count: ',len(neg_pairs[0])
        return neg_pairs,qd_mask


    def get_neg_skg_pair_batch(self,neg_pair_index, batch_size):
        if neg_pair_index + batch_size < len(self.neg_skg_pairs[0]):
            end_index = neg_pair_index + batch_size
        else:
            end_index = len(self.neg_skg_pairs[0])
            neg_pair_index = end_index - batch_size

        u_batch = self.neg_skg_pairs[0][neg_pair_index:end_index]
        v_batch = self.neg_skg_pairs[1][neg_pair_index:end_index]

        qd_mask = self.neg_skg_masks[neg_pair_index:end_index]

        return self.package_uv_batch(u_batch,v_batch,qd_mask)


    def package_uv_batch(self,u_batch,v_batch,qd_mask):
        u_lens = np.array([len(u) for u in u_batch])
        v_lens = np.array([len(v) for v in v_batch])

        #print 'u_lens:',u_lens,v_lens
        max_u_len = max(max(u_lens),3)
        max_v_len = max(max(v_lens),3)

        if self.use_bert:
            u_batch = np.array([self.pad_seq_bert(u, max_u_len) for u in u_batch])
            v_batch = np.array([self.pad_seq_bert(v, max_v_len) for v in v_batch])
        else:
            u_batch = np.array([self.pad_seq(u, max_u_len) for u in u_batch])
            v_batch = np.array([self.pad_seq(v, max_v_len) for v in v_batch])

        qd_mask = np.array(qd_mask)

        return (u_batch,v_batch,u_lens,v_lens,qd_mask)

    def skip_gram_reader(self,batch_size,is_loop=False):
        qd_mask = []
        u_batch, v_batch, max_u_len, max_v_len = [], [], 0, 0

        neg_pair_index = 0
        neg_pair_size = len(self.neg_skg_pairs)

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']
        max_seq_len = max(max_q_len,max_d_len)

        while True:
            for line in open(self.pair_addr):
                elements = line.strip().split('\t')
                u_node, v_node, u_text, v_text = elements

                if self.use_bert:
                    u_idx = bert_convert_ids(u_text, self.tokenizer)
                    v_idx = bert_convert_ids(v_text, self.tokenizer)
                    if len(u_idx) <= 2 or len(v_idx) <= 2:
                        continue
                else:
                    u_text = filter_title(u_text.split())
                    v_text = filter_title(v_text.split())
                    u_idx = map(lambda w: find_id(self.word2id, w), u_text)[:max_seq_len]
                    v_idx = map(lambda w: find_id(self.word2id, w), v_text)[:max_seq_len]

                    if len(u_idx) == 0 or len(v_idx) == 0:
                        continue

                u_mask = 0 if u_node[0] == 'd' else 1
                v_mask = 0 if v_node[0] == 'd' else 1
                qd_mask.append([u_mask, v_mask])

                u_batch.append(u_idx)
                v_batch.append(v_idx)

                max_u_len = max(max_u_len, len(u_idx))
                max_v_len = max(max_v_len, len(v_idx))

                if len(u_batch) >= batch_size:
                    pos_pair_batch = self.package_uv_batch(u_batch,v_batch,qd_mask)
                    neg_pair_batch = self.get_neg_skg_pair_batch(neg_pair_index,batch_size)

                    yield pos_pair_batch,neg_pair_batch

                    qd_mask = []
                    u_batch, v_batch, max_u_len, max_v_len = [], [], 0, 0
                    neg_pair_index = (neg_pair_index + batch_size) % neg_pair_size

            if is_loop == False:
                break

    def ranking_pair_reader(self,batch_size):

        click_model = self.config['click_model']
        model_id = model2id(click_model)
        text_list = os.listdir(self.train_rank_addr)

        query_batch, doc_pos_batch, doc_neg_batch, doc_pos_length, doc_neg_length = [],[],[],[],[];

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len,max_q_len)

        print 'text_list Size: ', len(text_list)
        epoch = 0
        last_time = time.time()

        pad_seq_ = self.pad_seq_bert if self.use_bert else self.pad_seq

        while True:
            random.shuffle(text_list)
            now = time.time()
            print ('\n\n----------Epoch: %d\t Time cost:%.4f----------\n\n' % (epoch, now - last_time))
            last_time = now
            epoch += 1

            for text_id in text_list:
                documents = [];relevances = [];lengths = [];
                for i,line in enumerate(open(os.path.join(self.train_rank_addr,text_id))):
                    if i == 0:
                        continue
                    elements = line.strip().split('\t')
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

                    labels = map(float,elements[-6:])
                    relevances.append(labels[model_id])
                    lengths.append(len(title_idx))

                if self.use_bert:
                    query_idx = bert_convert_ids(query, self.tokenizer)
                else:
                    query_idx = map(lambda w: find_id(self.word2id, w),filter_title(query.split()))[:max_seq_len]

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
                        doc_pos_batch.append(pos_doc)
                        doc_pos_length.append(pos_len)

                        doc_neg_batch.append(neg_doc)
                        doc_neg_length.append(neg_len)

                        if len(query_batch) >= batch_size:
                            query_lengths = np.array([len(s) for s in query_batch])
                            max_query_len = max(max(query_lengths),3)
                            max_pos_doc_len = max(max(doc_pos_length),3)
                            max_neg_doc_len = max(max(doc_neg_length),3)
                            #query_lengths = np.array(self.pad_seq(query_lengths,max_seq_len))
                            query_batch = np.array([pad_seq_(s, max_query_len) for s in query_batch])
                            doc_pos_batch = np.array([pad_seq_(d[:max_pos_doc_len], max_pos_doc_len) for d in doc_pos_batch])
                            doc_neg_batch = np.array([pad_seq_(d[:max_neg_doc_len], max_neg_doc_len) for d in doc_neg_batch])

                            yield (query_batch, query_lengths, doc_pos_batch,doc_pos_length, \
                                  doc_neg_batch, doc_neg_length)

                            query_batch, doc_pos_batch, doc_neg_batch, doc_pos_length, doc_neg_length = [], [], [], [], [];

    def ranking_point_reader(self,data_addr,is_test=False,label_type='PSCM'):
        model_id = model2id(label_type)
        text_list = os.listdir(data_addr)

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len, max_q_len)

        if 'test_mode' in self.config:
            text_list = text_list[:]

        for text_id in tqdm.tqdm(text_list[:]):
            doc_batch, gt_rels, doc_lengths = [], [], []
            dids = []
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

                index = -7 if is_test else -6
                labels = map(float,elements[index:])
                doc_batch.append(title_idx)
                doc_lengths.append(len(title_idx))
                gt_rels.append(labels[model_id])
                dids.append(did)

            if self.use_bert:
                query_idx = bert_convert_ids(query, self.tokenizer)
            else:
                query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))[:max_seq_len]
                query_idx = self.pad_seq(query_idx, max(max_q_len, 3))

            query_batch = [query_idx for i in range(len(doc_batch))]
            query_lengths = np.array([len(s) for s in query_batch])

            doc_lengths = np.array(doc_lengths)
            max_doc_len = max(max(doc_lengths),3)
            #print len(doc_batch),len(doc_batch[0]),max_doc_len
            doc_batch = np.array([self.pad_seq(d, max_doc_len) for d in doc_batch])


            #print 'doc_batch:',len(query_idx), doc_batch.shape, doc_lengths
            yield (qid,dids, query_batch, query_lengths, doc_batch, doc_lengths, gt_rels)

    def ntcir_data_loader(self,addr):
        data = defaultdict(lambda: defaultdict(list))
        for line in open(addr):
            elements = line.strip().split('\t')
            queryid = elements[0]
            query = elements[1]
            docid = elements[2]
            doc_content = elements[3]
            bm25 = float(elements[4])
            rel = 0
            # rel = float(int(elements[5]))
            # annotation = {docid:}  # q,d,r
            data[queryid][docid] = [query, doc_content, bm25, rel]

        max_q_len = self.config['max_q_len']
        max_d_len = self.config['max_d_len']

        max_seq_len = max(max_d_len, max_q_len)

        for qid in tqdm.tqdm(data.keys()):
            doc_batch, doc_lengths = [], []
            qids = [];dids = []
            for docId in test_data[qid].keys():
                elements = test_data[qid][docId]
                query = elements[0]
                doc_content = elements[1]
                bm25 = elements[2]
                gt = elements[3]


                title_words = filter_title(doc_content.split())
                title_idx = map(lambda w: find_id(self.word2id, w), title_words)[:max_seq_len]
                if len(title_idx) == 0:
                    continue

                qids.append(qid)
                dids.append(docId)

                doc_batch.append(title_idx)
                doc_lengths.append(len(title_idx))
            query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))[:max_seq_len]
            query_idx = self.pad_seq(query_idx, max(max_q_len, 3))
            item_size = len(doc_batch)
            query_batch = np.array([query_idx for i in range(item_size)])
            query_lengths = np.array([len(s) for s in query_batch])

            doc_lengths = np.array(doc_lengths)
            max_doc_len = max(max(doc_lengths), 3)
            doc_batch = np.array([self.pad_seq(d[:max_doc_len], max_doc_len) for d in doc_batch])

            # print 'query_batch:',query_batch.shape,doc_batch.shape
            yield (qids,dids,query_batch, doc_batch, query_lengths, doc_lengths)


    def pad_seq(self, seq, max_length, PAD_token=0):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    def pad_seq_bert(self, seq, max_length, PAD_token=103):
        # id of [MASK] is 103
        if len(seq) > max_length:
            seq = seq[:max_length - 1] + [seq[-1]]
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq