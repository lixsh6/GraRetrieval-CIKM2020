
from data.generator import *
from metrics.rank_evaluations import *
from tensorboardX import SummaryWriter
from ranker import *
from SkipGram import *
from embed import *
from config import *
import time

class EmbRanker(nn.Module):
    def __init__(self,args,config):
        super(EmbRanker, self).__init__()

        self.__dict__.update(config)
        self.ranker_name = args.encoder

        config['ranker'] = args.encoder


        self.data_generator = DataGenerator(config,args.m)
        self.embed_manager = Embed(self.data_generator.vocab_size, config)

        self.ranker = Ranker(config,self.embed_manager)
        self.skip_gram = SkipGram(self.embed_manager)

        if use_cuda:
            #torch.cuda.set_device(args.gpu)
            self.ranker.cuda()
            self.skip_gram.cuda()

        self.skgram_optimizer = getOptimizer(config['skgram_optim'],self.skip_gram.parameters(),
                                             lr=config['skgram_lr'])
        self.ranker_optimizer = getOptimizer(config['ranker_optim'], self.ranker.parameters(),
                                             lr=config['ranker_lr'])

        if args.resume:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, 'ranker_model.pkl'))
            self.ranker.load_state_dict(checkpoint['model_state_dict'])
            self.ranker_optimizer.load_state_dict(checkpoint['optimizer'])

            checkpoint = torch.load(os.path.join(args.resume, 'skgram_model.pkl'))
            self.skip_gram.load_state_dict(checkpoint['model_state_dict'])
            self.skgram_optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("Creating a new model")

        self.evaluator = rank_eval()

        self.message = args.m if args.m != '' else args.encoder
        self.writer = SummaryWriter(log_dir='./log/' + self.message, comment=self.message)

        print 'message: ', self.message

    def train(self):

        self.ranker.zero_grad()
        self.skip_gram.zero_grad()

        self.skgram_optimizer.zero_grad()
        self.ranker_optimizer.zero_grad()


        skg_Dataloader = self.data_generator.skip_gram_reader(self.batch_size,is_loop=False)

        skg_DataIterator = self.data_generator.skip_gram_reader(self.batch_size, is_loop=True)
        rank_DataIterator = self.data_generator.ranking_pair_reader(self.batch_size)

        start_time = time.time()
        '''
        skip gram pre-training step
        '''
        for epoch in range(self.pre_train_epoch):
            skg_loss_total = 0.
            for i,(pos_pair_batch, neg_pair_batch) in tqdm.tqdm(enumerate(skg_Dataloader)):
                skg_loss = self.train_skgram(pos_pair_batch, neg_pair_batch)
                skg_loss_total += skg_loss
                if i % self.train_freq == 0:
                    print ('Pre-training: Epoch%d\tStep:%d\tSkg_loss:%.3f\tElapsed:%.2f'\
                           % (epoch,i,skg_loss_total,time.time() - start_time))
                    skg_loss_total = 0.

        if self.pre_train_epoch > 0:
            self.save_checkpoint(0, 0, message=('pre_trained_model-%s' % self.ranker_name))

        '''
        jointly training for skip-gram and ranking
        '''

        #TODO: fine-tuning the lr of skgram
        total_rank_loss,total_skg_loss = 0.0,0.0
        patience = self.patience
        best_ndcg10 = 0.0
        last_ndcg10 = 0.0

        log_start_time = time.time()
        for step in xrange(self.steps):
            if self.train_skg_flag:
                pos_pair_batch, neg_pair_batch = next(skg_DataIterator)
                skg_loss = self.train_skgram(pos_pair_batch, neg_pair_batch)
            else:
                skg_loss = 0.0

            ranking_data = next(rank_DataIterator)
            rank_loss = self.train_ranker(ranking_data)

            total_skg_loss += skg_loss
            total_rank_loss += rank_loss

            if step % self.train_freq == 0 and step != 0:
                total_skg_loss /= self.train_freq
                total_rank_loss /= self.train_freq
                total_loss = total_skg_loss + total_rank_loss
                print ('Training: Step:%d\tTotal_loss:%.3f\tRank_loss:%.3f\tSkg_loss:%.3f\tElapsed:%.2f' \
                       % (step, total_loss,total_rank_loss,total_skg_loss, time.time() - start_time))

                loss_state = {'Total loss':total_loss,'Rank loss':total_rank_loss,'Skg loss':total_skg_loss}
                self.add_writer(loss_state,step,data_name='train')
                total_rank_loss, total_skg_loss = 0.0, 0.0

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

        print 'Train time: ',(log_start_time - time.time())
        print ("All done, exiting...")


    def train_skgram(self,pos_pair_batch, neg_pair_batch):
        self.skip_gram.train()
        self.skip_gram.zero_grad()
        self.skgram_optimizer.zero_grad()

        pos_u_batch, pos_v_batch, pos_u_lens, pos_v_lens, pos_qd_mask = pos_pair_batch
        neg_u_batch, neg_v_batch, neg_u_lens, neg_v_lens, neg_qd_mask = neg_pair_batch

        pos_pair,pos_lens = (pos_u_batch, pos_v_batch),(pos_u_lens, pos_v_lens)
        neg_pair,neg_lens = (neg_u_batch, neg_v_batch),(neg_u_lens, neg_v_lens)
        skg_loss = self.skip_gram(pos_pair, neg_pair,pos_lens,neg_lens, pos_qd_mask, neg_qd_mask)

        skg_loss = skg_loss
        skg_loss.backward()

        self.skgram_optimizer.step()

        return skg_loss.item()

    def train_ranker(self,ranking_data):
        self.ranker.train()
        self.ranker.zero_grad()
        self.ranker_optimizer.zero_grad()

        query_batch, query_lengths, doc_pos_batch, doc_pos_length, doc_neg_batch, doc_neg_length = ranking_data

        pos_score = self.ranker(query_batch, query_lengths, doc_pos_batch, doc_pos_length)
        neg_score = self.ranker(query_batch, query_lengths, doc_neg_batch, doc_neg_length)

        rank_loss = torch.sum(torch.clamp(1.0 - pos_score + neg_score,min = 0))

        rank_loss.backward()
        self.ranker_optimizer.step()

        return rank_loss.item()

    def test_ranker(self,data_addr,is_test=False,save_result=False,label_type='PSCM'):

        start_time = time.time()
        #visualize_file = open('./visualize/sort/%s.txt' % self.message, 'w')
        scores_list = {}
        with torch.no_grad():
            self.ranker.eval()
            results = defaultdict(list)

            index = 0
            for rank_data in self.data_generator.ranking_point_reader(data_addr,is_test=is_test,label_type=label_type):
                qid,dids, query_batch, query_lengths, doc_batch, doc_lengths, gt_rels = rank_data
                scores = self.ranker(query_batch, query_lengths, doc_batch, doc_lengths)

                scores = scores.cpu().numpy()

                gt_rels = map(lambda t: score2cutoff(label_type, t), gt_rels)
                result = self.evaluator.eval(gt_rels, scores)

                for k, v in result.items():
                    results[k].append(v)

                scores_list[index] = (qid, zip(list(dids), list(scores), list(gt_rels)), result['ndcg@1'])
                index += 1

        print 'message: ', self.message
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
            cPickle.dump(scores_list, open('./scores_list/%s.pkl'%self.message, 'w'))
            cPickle.dump(results,open(path,'w'))

        return performances

    def parse_as_line(self,qid,dids,cos_sims, query,doc_batch,scores,result,gts):
        query_words = [qid]
        for wid in query:
            if wid == 0:
                break
            query_words.append(self.data_generator.id2word[wid])

        doc_words = []
        for i in range(doc_batch.shape[0]):
            words = [dids[i],str(cos_sims[i])]
            for wid in doc_batch[i]:
                if wid == 0:
                    break
                words.append(self.data_generator.id2word[wid])
            doc_words.append(words)

        scores = scores.reshape(-1) * -1        #descending order
        sort_indices = np.argsort(scores)

        doc_words = np.array(doc_words)[sort_indices]
        gts = np.array(gts)[sort_indices]

        for i in range(len(doc_words)):
            doc_words[i].append(str(gts[i]))

        ndcg = str(result['ndcg@1']) + ' ' + str(result['ndcg@3']) + ' ' + str(result['ndcg@5'])
        out = ' '.join(query_words) + '\t' + ndcg + '\t'+ '\t'.join(map(lambda t:' '.join(t),doc_words))

        return out

    def save_checkpoint(self, step, best_ndcg10,message):
        filePath = os.path.join(self.saveModeladdr,message)
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        torch.save({
            'step': step,
            'model_state_dict': self.skip_gram.state_dict(),
            'best_result': best_ndcg10,
            'optimizer': self.skgram_optimizer.state_dict(),
        }, os.path.join(filePath, 'skgram_model.pkl'))

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
    train_model = EmbRanker(args,config)
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
