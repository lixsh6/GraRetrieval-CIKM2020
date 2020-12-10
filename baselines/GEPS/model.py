from utils import *
import cPickle


class GEPS(nn.Module):
    def __init__(self,config,vocab_size,query_size,doc_size):
        super(GEPS, self).__init__()

        self.embsize = config['embsize']
        self.kernel_num = config['kernel_num']#64
        self.kernel_size = config['kernel_size']#3
        self.embed_dim_graph = config['embed_dim_graph']

        self.term_embedding = nn.Embedding(vocab_size,self.embsize)
        self.queryGraph_embedding = nn.Embedding(query_size,self.embed_dim_graph)
        self.docGraph_embedding = nn.Embedding(doc_size, self.embed_dim_graph)


        self.load_embedding(self.term_embedding,config['emb'],name='word')
        self.load_embedding(self.queryGraph_embedding, config['query_emb'], name='query')
        self.load_embedding(self.docGraph_embedding, config['doc_emb'], name='doc')

        self.query_encoder = KimCNN(self.kernel_num,self.kernel_size,self.embsize,self.embed_dim_graph)
        self.doc_encoder = KimCNN(self.kernel_num, self.kernel_size, self.embsize, self.embed_dim_graph)

        self.query_linear = nn.Linear(self.embed_dim_graph,self.embed_dim_graph)
        self.out_linear = nn.Linear(self.embed_dim_graph * 3, 1)#query,doc, doc graph embed

    def load_embedding(self,embed,addr,name):
        print ('Loading %s embeddings' % name)
        pre_embeds = cPickle.load(open(addr))
        print (('pre_%s_embeds size: ' % name), pre_embeds.shape)
        embed.weight = nn.Parameter(torch.FloatTensor(pre_embeds))

    def forward(self,query_batch,doc_batch,docId_batch):
        query_var = Tensor2Varible(torch.LongTensor(query_batch))
        doc_var = Tensor2Varible(torch.LongTensor(doc_batch))
        doc_id_var = Tensor2Varible(torch.LongTensor(docId_batch))

        query_out = self.query_encoder(self.term_embedding(query_var))#(bs,dim)
        doc_out = self.doc_encoder(self.term_embedding(doc_var))

        doc_graph_out = self.docGraph_embedding(doc_id_var)

        out = torch.cat([query_out,doc_out,doc_graph_out],dim=-1)
        score = self.out_linear(out)

        return score, query_out


    def train_forward(self,query_batch,doc_batch,qId_batch,docId_batch):
        score, query_out = self.forward(query_batch,doc_batch,docId_batch)

        qid_var = Tensor2Varible(torch.LongTensor(qId_batch))
        query_out = self.query_linear(query_out)
        query_graph_embed = self.queryGraph_embedding(qid_var)#bs*dim

        query_loss = torch.sum((query_out.detach() - query_graph_embed)**2, dim=-1)

        return score, query_loss









