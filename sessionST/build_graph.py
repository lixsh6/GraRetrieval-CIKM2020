import networkx as nx
import random,tqdm
import numpy as np
import math,os,cPickle

from collections import Counter
import itertools,argparse
from config import *

def toqid(fname):
    return 'q' + fname.split('.')[0]

class GraphBuilder():
    def __init__(self,config):
        self.session_file = config['train_addr']
        self.path_length = config['path_length']
        self.graph_addr = config['graph_addr']
        self.query_dict_addr = config['query_dict']
        self.doc_dict_addr = config['doc_dict']

        self.coverge = config['converge']

        if os.path.exists(self.graph_addr):
            self.G = cPickle.load(open(self.graph_addr))
            print 'Number_of_nodes: ', self.G.number_of_nodes()
            print 'Number_of_edges: ', self.G.number_of_edges()
            print 'avg degree:',self.cal_degree()

            self.qid2text_dict = cPickle.load(open(self.query_dict_addr))
            self.did2text_dict = cPickle.load(open(self.doc_dict_addr))
            print 'Number of query: ', len(self.qid2text_dict)
            print 'Number of doc: ', len(self.did2text_dict)
        else:
            self.qid2text_dict = {}
            self.did2text_dict = {}
            self.G = self.build_graph()

        self.walks = []
        self.pairs = []


    def cal_degree(self):
        degrees = [val for (node, val) in self.G.degree() if node[0] == 'q']
        return np.mean(degrees)

    def build_graph(self,plot_graph=False):

        addr = self.session_file
        query_click_G = nx.Graph()
        session_folders = os.listdir(addr)
        print 'session count: ', len(session_folders)
        print 'building graph...'
        for i,sid_name in tqdm.tqdm(enumerate(session_folders)):
            s_addr = os.path.join(addr,sid_name)
            query_files = os.listdir(s_addr)
            query_files.sort()
            session_edge_list = []
            qids = map(lambda t:toqid(t.split('_')[1]),query_files)

            for i,qfile in enumerate(query_files):
                fname = os.path.join(s_addr,qfile)
                qc_edge_list = self.read_click_docs(fname)
                if len(qc_edge_list) > 0:
                    query_click_G.add_edges_from(qc_edge_list)
                if i > 0:
                    session_edge_list.append((qids[i-1],qids[i]))
            #print 'session_edge_list: ',session_edge_list
            query_click_G.add_edges_from(session_edge_list)

        print 'Number_of_nodes: ', query_click_G.number_of_nodes()
        print 'Number_of_edges: ', query_click_G.number_of_edges()
        print 'Number of query: ', len(self.qid2text_dict)
        print 'Number of doc: ', len(self.did2text_dict)

        print 'saving the graph as pkl...'
        cPickle.dump(query_click_G, open(self.graph_addr, 'w'))
        print 'saving the query and doc dict...'
        cPickle.dump(self.qid2text_dict, open(self.query_dict_addr, 'w'))
        cPickle.dump(self.did2text_dict, open(self.doc_dict_addr, 'w'))

        if plot_graph:
            # too big to draw
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            print 'saving the graph figure...'
            nx.draw(query_click_G)
            plt.savefig("query_click_G.png")

        return query_click_G

    def read_click_docs(self,filename):
        qc_edge_list = []
        for i,line in enumerate(open(filename)):
            elements = line.strip().split('\t')
            if i > 0 and elements[5] > 0.5: #skip first line and choose doc with click prob > 70% in the whole set
                qc_edge_list.append((elements[0],elements[1])) #(qid,did)
                qid = elements[0]
                did = elements[1]
                query = elements[2]
                title = elements[3]
            if i > 0:
                if qid not in self.qid2text_dict:
                    self.qid2text_dict[qid] = query
                if did not in self.did2text_dict:
                    self.did2text_dict[did] = title

        return qc_edge_list


    def id2text(self,pair_id):
        return self.qid2text_dict[pair_id] if pair_id[0] == 'q' else self.did2text_dict[pair_id]


def path_config():
    state = {}
    #input files
    state['train_addr'] = '../session/'
    #output files
    state['graph_addr'] = './data/graph/qc_graph.pkl'
    state['query_dict'] = './data/graph/query_dict.pkl'
    state['doc_dict'] = './data/graph/doc_dict.pkl'

    return state

def load_arguments():
    parser = argparse.ArgumentParser(description='Building the graph data and store as Pickle file')
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='path_config')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = load_arguments()
    config_state = eval(args.prototype)()
    builder = GraphBuilder(config_state)



