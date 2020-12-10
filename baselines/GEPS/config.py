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

    args = parser.parse_args()

    return args


def basic_config():
    state = {}
    state['min_score_diff'] = 0.25
    state['click_model'] = 'PSCM'
    state['train_skg_flag'] = True

    state['train_rank_addr'] = '../../ad-hoc-udoc/train/'
    state['valid_rank_addr'] = '../../ad-hoc-udoc/valid/'
    state['test_rank_addr'] = '../../ad-hoc-udoc/test/'

    state['local_data_addr'] = './data/'
    state['vocab_dict_file'] = '../../GraRanker/data/' + 'vocab.dict.9W.pkl'
    state['query_dict_file'] = state['local_data_addr'] + 'query.dict.pkl'
    state['doc_dict_file'] = state['local_data_addr'] + 'doc.dict.pkl'

    state['emb'] = '../../GraRanker/data/emb50_9W.pkl'
    state['query_emb'] = state['local_data_addr'] + 'query_embed.pkl'
    state['doc_emb'] = state['local_data_addr'] + 'doc_embed.pkl'

    state['saveModeladdr'] = './model/'

    state['patience'] = 5
    state['steps'] = 3000000         # Number of batches to process

    state['train_freq'] = 50  # 200
    state['eval_freq'] = 400  # 5000

    state['max_q_len'] = 15
    state['max_d_len'] = 20

    state['ranker_optim'] = 'adadelta' #adagrad
    state['ranker_lr'] = 0.01 #0.1

    state['out_size'] = 16
    state['drate'] = 0.8
    state['seed'] = 1234

    state['batch_size'] = 80
    #state['lr'] = 0.01#0.005
    state['weight_decay'] = 0#1e-3
    state['clip_grad'] = 0.5

    state['optim'] = 'adadelta'  # 'sgd, adadelta' adadelta0.1, adam0.005

    state['embsize'] = 50
    state['kernel_num'] = 64
    state['kernel_size'] = 3
    state['embed_dim_graph'] = 128

    state['cost_threshold'] = 1.003


    return state


def test_config():
    state = basic_config()

    state['train_freq'] = 10  # 200
    state['eval_freq'] = 20  # 5000

    state['test_mode'] = True
    state['steps'] = 30
    return state


def train_config():
    state = basic_config()
    return state



