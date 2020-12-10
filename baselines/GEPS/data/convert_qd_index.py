# encoding=utf8
'''
1. convert queryId and docId to embedding index
2. load pre-trained query and doc graph embedding
'''

import numpy as np
import cPickle

def loadEmbedding(addr):
    embed = {}
    for i,line in enumerate(open(addr)):
        if i == 0:
            continue
        elements = line.strip().split()
        idx = elements[0]
        vector = map(float,elements[1:])
        embed[idx] = vector
    return embed

first_order_embed = loadEmbedding('../../LINE/sogou-st/data/embedding_1order_64_session.txt')
second_order_embed = loadEmbedding('../../LINE/sogou-st/data/embedding_2order_64_session.txt')


qd_idx = first_order_embed.keys()

q2id,id2q = {'<unk>':0},{0:'<unk>'}
d2id,id2d = {'<unk>':0},{0:'<unk>'}

q_idx = [idx for idx in qd_idx if idx[0] == 'q']
d_idx = [idx for idx in qd_idx if idx[0] == 'd']

print 'query size: ', len(q_idx)
print 'doc size: ', len(d_idx)

q_embed = np.random.random((len(q_idx) + 1, 64 * 2))
d_embed = np.random.random((len(d_idx) + 1, 64 * 2))


for idx in q_idx:
    this_id = len(q2id) # start from 1
    q2id[idx] = this_id
    id2q[this_id] = idx
    q_embed[this_id] = np.array(first_order_embed[idx] + second_order_embed[idx])

for idx in d_idx:
    this_id = len(d2id)
    d2id[idx] = this_id
    id2d[this_id] = idx
    d_embed[this_id] = np.array(first_order_embed[idx] + second_order_embed[idx])

cPickle.dump((q2id,id2q),open('./query.dict.pkl','w'))
cPickle.dump((d2id,id2d),open('./doc.dict.pkl','w'))
cPickle.dump(q_embed,open('./query_embed.pkl','w'))
cPickle.dump(d_embed,open('./doc_embed.pkl','w'))