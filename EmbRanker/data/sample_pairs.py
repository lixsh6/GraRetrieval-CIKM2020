import cPickle
import numpy as np
percent = [0.2,0.4,0.6,0.8]

p_pair_addr = './random_walk/query_click/pair/converge10_walklen7_winsize3.txt'
neg_pair_addr = './random_walk/query_click/neg_pair/neg_pair_docratio_10.txt'
text_dict = {}

def load_pairID(addr):
    pairs = []
    for line in open(addr):
        elements = line.strip().split('\t')
        pairs.append((elements[0],elements[1]))
        if elements[0] not in text_dict:
            text_dict[elements[0]] = elements[2]
        if elements[1] not in text_dict:
            text_dict[elements[1]] = elements[3]
    return pairs

def find_text(nid):
    return text_dict[nid] if nid in text_dict else '<UNK>'

def filter_nodes(all_pairs,neg_pairs,select_nodes):
    p_pairs,n_pairs = [],[]
    for p,n in all_pairs:
        if p in select_nodes and n in select_nodes:
            p_pairs.append((p,n,find_text(p),find_text(n)))
    for p,n in neg_pairs:
        if p in select_nodes and n in select_nodes:
            n_pairs.append((p,n,find_text(p),find_text(n)))

    nodes = list(select_nodes)

    gap_number = len(p_pairs) - len(n_pairs)
    rp_nodes = np.random.choice(nodes, gap_number)
    rn_nodes = np.random.choice(nodes, gap_number)

    for p,n in zip(rp_nodes,rn_nodes):
        if p != n:
            n_pairs.append((p,n,find_text(p),find_text(n)))
    return p_pairs, n_pairs

def write(p_pairs, n_pairs, addr):
    p_file = open(addr + '_pos.txt','w')
    for elements in p_pairs:
        line = '\t'.join(elements)
        print >> p_file, line

    n_file = open(addr + '_neg.txt', 'w')
    for elements in n_pairs:
        line = '\t'.join(elements)
        print >> n_file, line

all_pairs = load_pairID(p_pair_addr)
neg_pairs = load_pairID(neg_pair_addr)


for p in percent:
    print 'percent: ',p
    select_nodes = cPickle.load(open('../../statistic/graph_size/sample_nodes_%.1f.pkl' % p))
    p_pairs, n_pairs = filter_nodes(all_pairs,neg_pairs,select_nodes)
    print 'p_pairs: ',len(p_pairs)
    print 'n_pairs: ',len(n_pairs)
    print '----------------------------'

    write(p_pairs, n_pairs, './graph_size/sample_%.1f' % p)
    #cPickle.dump((p_pairs, n_pairs), open('./graph_size/select_pn_pairs_%.1f.pkl','w'))

