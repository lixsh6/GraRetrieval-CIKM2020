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

    original_addr = '../sessionST'
    state['train_addr'] = original_addr + '/session/'

    state['query_dict'] = '../statistic/dict/query_dict.pkl'
    state['doc_dict'] = '../statistic/dict/doc_dict.pkl'

    state['vocab_dict_file'] = original_addr + '/GraRanker/data/vocab.dict.9W.pkl'
    state['emb'] = original_addr + '/GraRanker/data/emb50_9W.pkl'
    state['saveModeladdr'] = './model/'

    state['graph_type'] = 'query_click'
    state['depth'] = 3
    state['train_skg_pair_addr'] = ('./data/random_walk/%s/pair/converge10_walklen7_winsize%d.txt' % (state['graph_type'],state['depth']))
    state['neg_skg_pairs_addr'] = './data/random_walk/%s/neg_pair/neg_pair_docratio_10.txt' % state['graph_type']

    state['train_rank_addr'] = original_addr + '/dataset/train/'
    state['valid_rank_addr'] = original_addr + '/dataset/valid/'
    state['test_rank_addr'] = original_addr + '/dataset/test/'

    state['ntcir15_test'] = ''
    state['min_score_diff'] = 0.25
    state['click_model'] = 'PSCM'

    state['patience'] = 5
    state['steps'] = 3000000  # Number of batches to process
    state['pre_train_epoch'] = 2  # 5
    state['train_freq'] = 50  # 200
    state['eval_freq'] = 400  # 5000

    state['max_q_len'] = 15
    state['max_d_len'] = 20

    state['skgram_optim'] = 'adam'  # adagrad,adam
    state['ranker_optim'] = 'adam'
    state['skgram_lr'] = 0.00001 #* 0.5
    state['ranker_lr'] = 0.001

    state['batch_size'] = 80
    state['drate'] = 0.8
    state['seed'] = 1234
    state['clip_grad'] = 0.5

    state['embsize'] = 50
    state['text_dim'] = 50

    #state['ranker'] = 'ARCI' #ARCI, DSSM, LSTM-RNN
    state['cost_threshold'] = 1.003

    state['train_skg_flag'] = True
    return state


def path_config():
    state = basic_config()
    state['coverage'] = 10   #10
    state['path_length'] = 7
    state['window_size'] = 'NAN'

    return state


#-----------------------------------------------


def rank_only_config():
    state = basic_config()
    state['pre_train_epoch'] = 0
    state['train_skg_flag'] = False
    state['skgram_optim'] = 'adam'  # adagrad,adam
    state['ranker_optim'] = 'adam'
    state['lr'] = 0.001
    return state

def test_config():
    state = basic_config()

    state['pre_train_epoch'] = 0

    state['train_freq'] = 10  # 200
    state['eval_freq'] = 5  # 5000

    state['test_mode'] = True
    state['steps'] = 50
    return state

def pretrain_config():
    state = basic_config()
    state['pre_train_epoch'] = 2
    state['skgram_lr'] = 0.001 * 0.2

    return state


def train_config():
    state = basic_config()
    state['pre_train_epoch'] = 0
    state['momentum'] = 0.9
    return state

def graph_size_config():
    state = basic_config()
    ratio = 0.2
    state['train_skg_pair_addr'] = './data/graph_size/sample_%.1f_pos.txt' % ratio
    state['neg_skg_pairs_addr'] = './data/graph_size/sample_%.1f_neg.txt' % ratio

    return state


def bert_config():
    state = basic_config()
    state['BERT_folder'] = '../bert/trained_model/bert-base-chinese'  # your path for model and vocab
    state['BERT_VOCAB'] = 'bert-base-chinese-vocab.txt'
    state['text_dim'] = 768
    state['bert_id2textid_dict'] = '../statistic/dict/bert_id2textid_dict.pkl'
    state['output_layer_index'] = -1
    state['batch_size'] = 10
    state['lr'] = 2e-5
    return state

