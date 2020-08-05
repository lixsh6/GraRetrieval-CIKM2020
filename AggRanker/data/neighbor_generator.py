import networkx as nx
import argparse,cPickle,tqdm

import numpy as np

def generate(graph_addr,save_addr):
    print 'graph: ',graph_addr
    graph = cPickle.load(open(graph_addr))
    print graph.number_of_nodes()
    nodes = [node for node in list(graph.nodes)]
    neighbor_dict = {}
    n_size = []
    for node in tqdm.tqdm(nodes):
        neighs = [nid for nid in graph.neighbors(node)]
        neighbor_dict[node] = neighs
        n_size.append(len(neighs))
    print 'avg neigh size: ',np.mean(n_size)
    cPickle.dump(neighbor_dict,open(save_addr,'w'))

def main():
    graph_type = 'query_doc_by_term'
    graph_addr = '../../statistic/graph/%s.pkl' % graph_type #query_click.pkl,click.pkl,query.pkl,query_doc_by_term.pkl
    save_addr = './neighbor_dict-%s.pkl' % graph_type
    generate(graph_addr, save_addr)

if __name__ == '__main__':
    main()