import cPickle

neighbor_dict_all = cPickle.load(open('./neighbor_dict-query_click.pkl'))


def sample_neib_dict(p):
    new_dict = {}
    select_nodes = cPickle.load(open('../../statistic/graph_size/sample_nodes_%.1f.pkl' % p))
    for key,neibs in neighbor_dict_all.iteritems():
        if key not in select_nodes:
            continue
        nei = [nid for nid in neibs if nid in select_nodes]
        new_dict[key] = nei
    print 'size:',len(new_dict)
    return new_dict

percent = [0.2,0.4,0.6,0.8]
for p in percent:
    print 'percent: ', p
    new_dict = sample_neib_dict(p)
    cPickle.dump(new_dict, open('./graph_size/neighbor_dict-query_click-%.1f.pkl' % p,'w'))
