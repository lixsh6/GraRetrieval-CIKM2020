
from data.generator import *
from metrics.rank_evaluations import *
from tensorboardX import SummaryWriter
from model import *
from config import *
import time


class TrainModel(nn.Module):
    def __init__(self,args,config):
        super(TrainModel, self).__init__()

        self.__dict__.update(config)
        self.data_generator = DataGenerator(config)

        vocab_size = self.data_generator.vocab_size
        query_size = self.data_generator.query_size
        doc_size = self.data_generator.doc_size
        self.ranker = GEPS(config,vocab_size,query_size,doc_size)

        if use_cuda:
            torch.cuda.set_device(args.gpu)
            self.ranker.cuda()

        self.ranker_optimizer = getOptimizer(config['ranker_optim'], self.ranker.parameters(),
                                             lr=config['ranker_lr'])
        self.evaluator = rank_eval()

        self.message = args.m
        self.writer = SummaryWriter(log_dir='./log/' + self.message, comment=self.message)

    def train(self):

        self.ranker.zero_grad()
        self.ranker_optimizer.zero_grad()
        rank_DataIterator = self.data_generator.ranking_pair_reader(self.batch_size)

        start_time = time.time()

        total_loss = 0.0
        total_rank_loss = 0.0
        total_q_loss = 0.0

        patience = self.patience
        best_ndcg10 = 0.0
        last_ndcg10 = 0.0

        for step in xrange(self.steps):
            ranking_data = next(rank_DataIterator)
            loss,rank_loss,q_loss = self.train_ranker(ranking_data)

            total_loss += loss
            total_rank_loss += rank_loss
            total_q_loss += q_loss

            if step % self.train_freq == 0 and step != 0:
                total_rank_loss /= self.train_freq
                print ('Training: Step:%d\tTotal_loss:%.3f\tRank_loss:%.3f\tQuery_loss:%.3f\tElapsed:%.2f' \
                       % (step, total_loss,total_rank_loss,total_q_loss, time.time() - start_time))

                loss_state = {'Total loss':total_loss,'Rank loss':total_rank_loss,'Query loss':total_q_loss}
                self.add_writer(loss_state,step,data_name='train')
                total_loss, total_rank_loss, total_q_loss = 0.0, 0.0, 0.0

            if step % self.eval_freq == 0 and step != 0:
                valid_performance= self.test_ranker(self.valid_rank_addr)
                current_ndcg10 = valid_performance['ndcg@10']

                if current_ndcg10 > best_ndcg10:
                    print '----Got better result, save to %s' % self.saveModeladdr
                    best_ndcg10 = current_ndcg10
                    patience = self.patience
                    self.save_checkpoint(step,best_ndcg10,message=self.message)
                elif current_ndcg10 <= last_ndcg10 * self.cost_threshold:
                    patience -= 1
                last_ndcg10 = current_ndcg10
                print 'Patience: ',patience
                print '------------------------------------'
                self.add_writer(valid_performance, step, data_name='valid')

            if patience < 0:
                break

        print ("All done, exiting...")

    def train_ranker(self,ranking_data):
        self.ranker.train()
        self.ranker.zero_grad()
        self.ranker_optimizer.zero_grad()

        query_batch, doc_pos_batch, doc_neg_batch, query_id_batch, docid_pos_batch, docid_neg_batch = ranking_data

        pos_score,pos_q_loss = self.ranker.train_forward(query_batch, doc_pos_batch, query_id_batch, docid_pos_batch)
        neg_score,neg_q_loss = self.ranker.train_forward(query_batch, doc_neg_batch, query_id_batch, docid_neg_batch)

        rank_loss = torch.sum(torch.clamp(1.0 - pos_score + neg_score,min = 0))
        q_loss = torch.sum(pos_q_loss + neg_q_loss)

        loss = rank_loss + q_loss
        loss.backward()

        self.ranker_optimizer.step()

        return loss.item(), rank_loss.item(), q_loss.item()

    def test_ranker(self,data_addr,is_test=False,save_result=False,label_type='PSCM'):
        with torch.no_grad():
            self.ranker.eval()
            results = defaultdict(list)

            for rank_data in self.data_generator.ranking_point_reader(data_addr,is_test=is_test,label_type=label_type):
                query_batch, doc_batch, docid_batch, gt_rels = rank_data
                scores, _ = self.ranker(query_batch, doc_batch, docid_batch)
                scores = scores.cpu().numpy()

                gt_rels = map(lambda t: score2cutoff(label_type, t), gt_rels)
                result = self.evaluator.eval(gt_rels, scores)

                for k, v in result.items():
                    results[k].append(v)

        performances = {}
        for k, v in results.items():
            performances[k] = np.mean(v)

        print '-----------------------------Performance:-----------------------------'
        print 'Data addr: ', data_addr
        print performances

        if save_result:
            path = './results/' + self.message + '_' + label_type + '_result.pkl'
            cPickle.dump(results,open(path,'w'))

        return performances

    def save_checkpoint(self, step, best_ndcg10,message):
        filePath = os.path.join(self.saveModeladdr,message)
        if not os.path.exists(filePath):
            os.makedirs(filePath)

        torch.save({
            'step': step,
            'model_state_dict': self.ranker.state_dict(),
            'best_result': best_ndcg10,
            'optimizer': self.ranker_optimizer.state_dict(),
        }, os.path.join(filePath, 'ranker_model.pkl'))

        return

    def add_writer(self, performance, step, data_name='vaild'):
        for metric, value in performance.iteritems():
            self.writer.add_scalar(data_name + '/' + metric, value, step)


def main(args,config):
    train_model = TrainModel(args,config)
    if not args.eval:
        train_model.train()
    else:
        print 'only evaluation'

    train_model.test_ranker(config['test_rank_addr'],is_test=True,save_result=True,label_type='PSCM')
    train_model.test_ranker(config['test_rank_addr'], is_test=True, save_result=True, label_type='DBN')
    train_model.test_ranker(config['test_rank_addr'], is_test=True, save_result=True, label_type='HUMAN')
    train_model.writer.close()

if __name__ == '__main__':
    args = load_arguments()
    config_state = eval(args.prototype)()
    main(args, config_state)
