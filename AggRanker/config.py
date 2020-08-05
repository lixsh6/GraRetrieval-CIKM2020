import argparse

def load_arguments():
    parser = argparse.ArgumentParser(description='Evidence Reading Model')

    parser.add_argument('--resume', type=str, default="",
                        help='Resume training from that state')
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='basic_config')
    parser.add_argument("--eval", action="store_true", help="only evaluation")

    parser.add_argument('--gpu', type=int, default=1,
                        help="# of GPU running on")

    parser.add_argument('--m', type=str, default="",
                        help='Message in visualizationn window (Flag)')

    parser.add_argument('-e', '--encoder',type=str,default="ACRI", help="The basic ranker")
    args = parser.parse_args()

    return args

def basic_config():
    state = {}

    original_addr = './sessionST'
    state['train_addr'] = original_addr + '/session/'

    state['query_dict'] = '../statistic/dict/query_dict.pkl'
    state['doc_dict'] = '../statistic/dict/doc_dict.pkl'

    state['vocab_dict_file'] = original_addr + '/GraRanker/data/vocab.dict.9W.pkl'
    state['emb'] = original_addr + '/GraRanker/data/emb50_9W.pkl'
    state['saveModeladdr'] = './model/'

    state['train_rank_addr'] = original_addr + '/dataset/train/'
    state['valid_rank_addr'] = original_addr + '/dataset/valid/'
    state['test_rank_addr'] = original_addr + '/dataset/test/'

    state['graph_type'] = 'query_click'
    state['query_dict_addr'] = '../statistic/dict/qid2textid_dict.pkl'
    state['doc_dict_addr'] = '../statistic/dict/did2textid_dict.pkl'
    state['neib_dict_addr'] = './data/neighbor_dict-%s.pkl' % state['graph_type']#query,click,query_doc_by_term


    state['min_score_diff'] = 0.25
    state['click_model'] = 'PSCM'

    state['patience'] = 5
    state['steps'] = 3000000  # Number of batches to process
    state['pre_train_epoch'] = 2  # 5
    state['train_freq'] = 50  # 200
    state['eval_freq'] = 400  # 5000

    state['max_q_len'] = 15
    state['max_d_len'] = 20

    state['aggregation_type'] = 'GraphSage'#GraphSage
    state['ranker_optim'] = 'adam'  # adagrad,adam,adadelta
    state['momentum'] = 0
    state['ranker_lr'] = 0.001
    state['weight_decay'] = 0

    state['batch_size'] = 80
    state['drate'] = 0.8
    state['seed'] = 1234
    state['clip_grad'] = 0.5

    state['embsize'] = 50
    state['text_dim'] = 50

    #state['ranker'] = 'ARCI' #ARCI, DSSM, LSTM-RNN
    state['cost_threshold'] = 1.003
    state['train_skg_flag'] = True

    state['test_mode'] = False
    return state


def path_config():
    state = basic_config()
    state['coverage'] = 10   #10
    state['path_length'] = 7
    state['window_size'] = 'NAN'

    return state

#-----------------------------------------------

def train_config():
    state = basic_config()
    return state

def bert_config():
    state = basic_config()
    state['BERT_folder'] = '../bert/trained_model/bert-base-chinese'  # your path for model and vocab
    state['BERT_VOCAB'] = 'bert-base-chinese-vocab.txt'
    state['text_dim'] = 768
    state['bert_id2textid_dict'] = '../statistic/dict/bert_id2textid_dict.pkl'
    state['output_layer_index'] = 2
    state['batch_size'] = 10
    state['lr'] = 2e-5
    return state

def bert_test_config():
    state = bert_config()
    state['train_freq'] = 10  # 200
    state['eval_freq'] = 5  # 5000
    state['test_mode'] = True
    state['steps'] = 100
    return state

def test_config():
    state = basic_config()

    state['train_freq'] = 10  # 200
    state['eval_freq'] = 20  # 5000

    state['test_mode'] = True
    state['steps'] = 30
    return state

def graph_size_config():
    state = basic_config()
    #state['resume'] = './model/ARCI-GraphSage-raw_drop1/'
    ratio = 0.4
    state['neib_dict_addr'] = './data/graph_size/neighbor_dict-query_click-%.1f.pkl' % ratio

    return state
