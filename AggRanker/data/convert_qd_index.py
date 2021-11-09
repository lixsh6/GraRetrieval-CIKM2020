# encoding=utf8
'''
1. convert queryId and docId to embedding index
'''
import cPickle
vocab_dict_file = './vocab.dict.9W.pkl'
query_dict_addr = './random_walk/graph/query_dict.pkl'
doc_dict_addr = './random_walk/graph/doc_dict.pkl'

n2id,id2n,id2wordid = {'<unk>':0},{0:'<unk>'},{0:0}
word2id, id2word = cPickle.load(open(vocab_dict_file))

qid2text_dict = cPickle.load(open(query_dict_addr))
did2text_dict = cPickle.load(open(doc_dict_addr))


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


for qid,query in qid2text_dict.iteritems():
    index = len(n2id)
    n2id[qid] = index
    id2n[index] = qid
    id2wordid[index] = map(lambda w: find_id(word2id, w), filter_title(query.split()))
    if index == 2:
        print id2wordid[2]


for did,doc in did2text_dict.iteritems():
    index = len(n2id)
    n2id[did] = index
    id2n[index] = did
    id2wordid[index] = map(lambda w: find_id(word2id, w), filter_title(doc.split()))

print 'size: ', len(n2id)
cPickle.dump((n2id,id2n,id2wordid),open('./node.dict.pkl','w'))

