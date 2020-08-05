from collections import defaultdict
import tqdm,os,sys,cPickle
from scipy.stats import ranksums,ttest_ind
import numpy as np

sys.path.append('../')

target_addr = '../../reader-qcl/results'
compared_addr = '../results'

target_file = '../../EmbRanker/results/transform-base-nodrop-3layer-adam1e-3-output-sum-pos_result.pkl'
#compared_file = '../results/transform-aggranker-dout0.2-dmid0.2-din0_result.pkl'
compared_file = '../../EmbRanker/results/transform-embranker-sum-pos-2-dout0.2_result.pkl'

target_result = cPickle.load(open(target_file))
compared_result = cPickle.load(open(compared_file))
for metric in compared_result.keys():
    print '--------------%s-------------------' % metric
    print 'compared: %.4f' % np.mean(compared_result[metric])
    print 'target: %.4f' % np.mean(target_result[metric])
    print 'T-test: ', ttest_ind(compared_result[metric],target_result[metric])