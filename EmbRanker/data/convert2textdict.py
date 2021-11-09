# encoding=utf8
'''
Convert text to dict
'''

import logging
import itertools
import cPickle, os, re, tqdm,sys
import numpy as np
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text2dict')
reload(sys)
sys.setdefaultencoding('utf8')

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("queryFile", type=str,
#                     help="queries file")
# parser.add_argument("docFile", type=str,
#                     help="documents file")
parser.add_argument("--cutoff", type=int, default=-1,
                    help="Vocabulary cutoff (optional)")
parser.add_argument("--min_freq", type=int, default=1,
                    help="Min frequency cutoff (optional)")
parser.add_argument("--dict", type=str, default="",
                    help="External dictionary (pkl file)")
# parser.add_argument("output", type=str, help="Output file")

parser.add_argument('-m', '--mapper', action='store_true', help='mapper')
parser.add_argument('-r', '--reducer', action='store_true', help='reducer')

args = parser.parse_args()

useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・']

def docParse(content):
    # word_count = content.split()
    document = []
    '-——_【】().,《》?、（）。:，'
    sents = re.split(r"--\s|[-——_【】().,《》?、（）。:，.,;?!\t\n\r]", content)
    for sent in sents:
        words = sent.strip().split()
        sent_new = []
        for word in words:
            w_new = word
            if len(w_new) != 0:
                sent_new.append(w_new)
        if len(sent_new) > 1:
            document.append(sent_new)

    return document


def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)
    return words

def generateDict(data_addr,save_addr):

    print 'Constructing dict'
    if args.dict != "":
        # Load external dictionary
        addr = save_addr + args.dict #vocab.dict.pkl
        assert os.path.isfile(addr)
        (vocab, id2word) = cPickle.load(open(addr, "r"))
        # vocab = dict([(x[0], x[1]) for x in cPickle.load(open(args.dict, "r"))])
        # Check consistency
        assert '<unk>' in vocab
        assert '<m>' in vocab  # mask
    else:
        word_counter = Counter()

        text_list = os.listdir(data_addr)
        for i in tqdm.tqdm(range(0, len(text_list))):
            filepath = os.path.join(data_addr, text_list[i])

            for line in open(filepath):
                elements = line.strip().split('\t')

                query_terms = elements[2].split()
                doc_words = elements[3].split()


                word_counter.update(filter_title(doc_words))
            word_counter.update(query_terms)


        total_freq = sum(word_counter.values())
        logger.info("Total word frequency in dictionary %d " % total_freq)

        if args.cutoff != -1:
            logger.info("Cutoff %d" % args.cutoff)
            vocab_count = word_counter.most_common(args.cutoff)
        else:
            vocab_count = word_counter.most_common()

        vocab = {'<m>': 0, '<unk>': 1}
        id2word = {0: '<m>', 1: '<unk>'}

        for (word, count) in vocab_count:
            if count < args.min_freq:
                break
            this_id = len(vocab)
            vocab[word] = this_id
            id2word[this_id] = word

        cPickle.dump((vocab, id2word), open(save_addr + 'vocab.dict.pkl', 'w'))
        print 'Dict Constructed'
        logger.info("Vocab size %d" % len(vocab))
    return vocab, id2word


def saveEmbedding(word_dict,model,save_addr,embedding_size=50):
    emb = np.random.uniform(low=-1, high=1, size=(len(word_dict), embedding_size))
    for word,wid in tqdm.tqdm(word_dict.iteritems()):
        if word in model:
            emb[wid,:] = model[word]

    print 'Saving embeddings'
    cPickle.dump(emb,open(save_addr + 'emb50.pkl','w'))
    #return emb

def load_options():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args, parser

if __name__ == "__main__":

    data_addr =  '../../ad-hoc-udoc/'
    save_addr = './'

    word_dict, id2word = generateDict(data_addr, save_addr)


    import gensim

    vector_addr = '~/qcl/reader-qcl/data/data/wiki.zh.text.50.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_addr, binary=False)

    #save_addr = './data/data/'
    saveEmbedding(word_dict, model, save_addr)

    '''

    import gensim

    print 'Loading vectors'
    #vector_addr = '/mnt/work1/lixin/data/word_embedding/wiki.en.vec'
    vector_addr = '/mnt/work/lixin/wiki2vec/data/dim50/wiki.en.text.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_addr, binary=False)
    print 'Loaded'

    for i in tqdm.tqdm(range(1,6)):
        qrels_addr = './MQ-data/data/%s/qrels/Fold%d/' % (dataName, i)
        queryFile = './MQ-data/data/%s/query/%s_query_dict.txt' % (dataName, dataName)
        word_dict, id2word = generateDict(queryFile, qrels_addr)
        saveEmbedding(word_dict,model,qrels_addr)

        #filetypes = ['train', 'vali', 'test']
        #for v in filetypes:
        #    generatePairFile(word_dict, qrels_addr, filetype=v)
    '''




