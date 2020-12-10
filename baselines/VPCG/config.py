import argparse


def load_arguments():
    parser = argparse.ArgumentParser(description='VPCG Model (SIGIR16)')

    parser.add_argument("--prototype", type=str, help="Use the prototype", default='basic_config')

    args = parser.parse_args()

    return args

def basic_config():
    state = {}
    state['train_addr'] = '../../ad-hoc-udoc/train/'
    state['valid_addr'] = '../../ad-hoc-udoc/valid/'
    state['test_addr'] = '../../ad-hoc-udoc/test/'

    state['qd_embed_addr'] = './data/qd_embed.pkl'
    state['word_embed_addr'] = './data/word_embed.pkl'
    state['local_data_addr'] = '../../GraRanker/data/'
    state['vocab_dict_file'] = state['local_data_addr'] + 'vocab.dict.9W.pkl'

    state['graph_addr'] = './data/graph.pkl'

    state['topK'] = 20 #top 20 words to be considered
    state['iteration'] = 100

    return state