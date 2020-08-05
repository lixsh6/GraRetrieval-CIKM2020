
from data.generator import *
from metrics.rank_evaluations import *
from tensorboardX import SummaryWriter
from ranker import *
from embed import *
from config import *
import time,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class TrainModel(nn.Module):
    def __init__(self,args,config):
        super(TrainModel, self).__init__()

        self.__dict__.update(config)
        self.ranker_name = args.encoder
        config['ranker'] = args.encoder

        self.data_generator = DataGenerator(config,args.m)
        self.embed_manager = Embed(self.data_generator.vocab_size,config)

        self.ranker = Ranker(config,self.embed_manager)

        if use_cuda:
            #torch.cuda.set_device(args.gpu)
            self.embed_manager.cuda()
            self.ranker.cuda()


        self.ranker_optimizer = getOptimizer(config['ranker_optim'], self.ranker.parameters(),
                                             lr=config['ranker_lr'],weight_decay=config['weight_decay'])#momentum=config['momentum']

        if args.resume:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, 'ranker_model.pkl'))

            self.ranker.load_state_dict(checkpoint['model_state_dict'])
            self.ranker_optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("Creating a new model")

        self.evaluator = rank_eval()

        self.message = args.m
        self.writer = SummaryWriter(log_dir='./log/' + self.message, comment=self.message)
        print 'message: ', self.message

    def train(self):

        self.ranker.zero_grad()
        self.ranker_optimizer.zero_grad()

        rank_DataIterator = self.data_generator.ranking_pair_reader(self.batch_size)

        start_time = time.time()

        total_rank_loss = 0.0
        patience = self.patience
        best_ndcg10 = 0.0
        last_ndcg10 = 0.0

        log_start_time = time.time()
        time_cost_list = []
        for step in xrange(self.steps):
            ranking_data = next(rank_DataIterator)
            rank_loss = self.train_ranker(ranking_data)
            total_rank_loss += rank_loss

            if step % self.train_freq == 0 and step != 0:
                total_rank_loss /= self.train_freq
                now_time = time.time()
                time_cost = now_time - start_time
                time_cost_list.append(time_cost)
                print ('Training: Step:%d\tRank_loss:%.3f\tElapsed:%.2f\tAvg Cost:%.2f' \
                       % (step,total_rank_loss, time_cost,np.mean(time_cost_list)))
                start_time = now_time

                loss_state = {'Rank loss':total_rank_loss}
                self.add_writer(loss_state,step,data_name='train')
                total_rank_loss = 0.0

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

        print 'Train time: ', (log_start_time - time.time())
        print ("All done, exiting...")


    def train_ranker(self,ranking_data):
        self.ranker.train()
        self.ranker.zero_grad()
        self.ranker_optimizer.zero_grad()

        query_batch, doc_pos_batch, doc_neg_batch, query_lengths, doc_pos_length, doc_neg_length, \
        query_ids, doc_pos_ids, doc_neg_ids = ranking_data

        pos_score = self.ranker(query_ids, doc_pos_ids, query_batch,doc_pos_batch, query_lengths, doc_pos_length)
        neg_score = self.ranker(query_ids, doc_neg_ids, query_batch,doc_neg_batch, query_lengths, doc_neg_length)

        rank_loss = torch.sum(torch.clamp(1.0 - pos_score + neg_score,min = 0))

        rank_loss.backward()
        self.ranker_optimizer.step()

        return rank_loss.item()

    def test_ranker(self,data_addr,is_test=False,save_result=False,label_type='PSCM'):

        start_time = time.time()
        print 'message: ', self.message
        scores_list = {}
        with torch.no_grad():
            self.ranker.eval()
            results = defaultdict(list)
            index = 0
            error = 0
            for rank_data in self.data_generator.ranking_point_reader(data_addr,is_test=is_test,label_type=label_type):
                query_ids, doc_ids, query_batch, doc_batch, query_lengths, doc_lengths, gt_rels = rank_data
                try:
                    scores = self.ranker(query_ids,doc_ids,query_batch, doc_batch, query_lengths, doc_lengths)
                except Exception as e:
                    print 'error: ', error
                    print e
                    error += 1
                scores = scores.view(-1).cpu().numpy()
                gt_rels = map(lambda t: score2cutoff(label_type, t), gt_rels)

                result = self.evaluator.eval(gt_rels, scores)

                for k, v in result.items():
                    results[k].append(v)

                scores_list[index] = (query_ids[0], zip(list(doc_ids), list(scores), list(gt_rels)), result['ndcg@1'])
                index += 1


        print('%s \t Test time: %.2f' % (label_type, time.time() - start_time))

        performances = {}
        for k, v in results.items():
            performances[k] = np.mean(v)

        print '-----------------------------Performance:-----------------------------'
        print 'Label: ',label_type
        print 'Data addr: ', data_addr
        print performances

        if save_result:
            path = './results/' + self.message + '_result.pkl'
            cPickle.dump(scores_list, open('./scores_list/%s.pkl' % self.message, 'w'))
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


    #train_model.test_ranker(config['test_rank_addr'],is_test=True,save_result=True,label_type='PSCM')
    #train_model.test_ranker(config['test_rank_addr'], is_test=True, save_result=True, label_type='DBN')
    train_model.test_ranker(config['test_rank_addr'], is_test=True, save_result=True, label_type='HUMAN')

    train_model.writer.close()

if __name__ == '__main__':
    args = load_arguments()
    config_state = eval(args.prototype)()

    main(args, config_state)
