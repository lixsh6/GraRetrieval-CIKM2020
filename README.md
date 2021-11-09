A pytorch implement for CIKM 2020 paper ''[Learning Better Representations for Neural Information Retrieval with Graph Information](http://www.thuir.cn/group/~YQLiu/)'', namely `Embedding-based neural ranker (EmbRanker)` and `Aggregation-based neural ranker (AggRanker)`.

## Requirement
* Python 2.7
* Pytorch 0.4.1
* tqdm
* networkx 2.1

## Dataset
We run experiment on the publicly available dataset [Tiangong-ST](http://www.thuir.cn/tiangong-st/), which is a Chinese search log from [Sogou.com](sogou.com). 

*	Preprocessed data should be placed in `./sessionST/dataset/`, following the settings in `config.py`. 
* Sampled files are given in `valid/test` folders. Each line consists of `qid	docid	query	title	TACM	PSCM	THCM	UBM	DBN	POM	HUMAN(Only available in test set)`, separated by `TAB`. In particular, `TACM	PSCM	THCM	UBM	DBN	POM` are the click labels given in the dataset.
* Building the graph data from session data requires `networkx` and `cPickle`. The graph data is stored as `pkl` file. Demo processing code is shown in [build_graph.py](./sessionST/build_graph.py).
* Run `./EmbRanker/data/convert2textdict.py` to create `vocab_dict_file` and embedding dict `emb` file. Embedding is downloaded from [here](http://download.wikipedia.com/zhwiki) .
* 

Besides, constructing the training-specific graph data in both models are different:

* **EmbRanker**: run `path_generator.py` based on pkl graph data to get positive and negative samples.
* **AggRanker**: run `./data/neighbor_generator.py` based on pkl graph data to get neighbors of each center nodes. 

## Baselines

The baseline code is released through our `PyTorch` implementation. 

1. `VPCG` is " [**Learning Query and Document Relevance from a Web-scale Click Graph**](http://www.yichang-cs.com/yahoo/SIGIR16_clickgraph.pdf) " (SIGIR 2016)
2. `GEPS` is "[**Neural IR Meets Graph Embedding: A Ranking Model for Product Search**](https://arxiv.org/pdf/1901.08286.pdf)" (WWW 2019)

## Procedure

1. All the settings are in `config.py`.
2. run `python main.py --prototype train_config -e ACRI --gpu 0`

----------------

If you have any problems, please contact me via `lixsh6@gmail.com`.

