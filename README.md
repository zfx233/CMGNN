# CMGNN
This is a unified codebase for heterophilous graph neural networks, including 13 baseline methods, one novel method CMGNN, and 10 datasets. 

## Baseline Methods
* **MLP**: Multilayer perceptron
* **GCN**: Semi-supervised Classification with Graph Convolutional Networks (ICLR 2017)
* **GAT**: Graph Attention Networks (ICLR, 2018)
* **GCNII**: Simple and Deep Graph Convolutional Networks (ICML 2020)
* **MixHop**: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing (ICML 2019)
* **H2GCN**: Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs (NeurIPS 2020)
* **GBKGNN**: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily (WWW 2022)
* **GGCN**: Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks (ICDM 2022)
* **GloGNN**: Finding Global Homophily in Graph Neural Networks When Meeting Heterophily (ICML 2022)
* **HOGGCN**: Powerful Graph Convolutioal Networks with Adaptive Propagation Mechanism for Homophily and Heterophily (AAAI 2022)
* **GPRGNN**: Adaptive Universal Generalized PageRank Graph Neural Network (ICLR 2021)
* **ACMGCN**: Revisiting Heterophily for Graph Neural Networks (NeurIPS 2022)
* **OrderedGNN**: Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-smoothing (ICLR 2023)


## Datasets
The datasets used in the codebase include **Roman-Empire**, **Amazon-Ratings**, **Chameleon-F**, **Squirrel-F**, **Actor**, **Flickr**, **BlogCatalog**, **Wikics**, **Pubmed**, and **Photo**.




## How to Run

### Baseline Methods
The presearched parameters are listed in '/config/baseline/{model_type}.yaml'. 

    python main.py --model_type={model_type} --dataset={dataset}

Note the dataset name is the script should use plain lowercase letters, e.g. for Chameleon-F dataset with "--dataset=chameleonf".

### Compatibility Matrix-aware GNN (CMGNN)

    python main.py --model_type=CMGNN --dataset={dataset}


## Main Requirements
* python >= 3.6.13
* numpy >= 1.19.2
* pytorch >= 1.10.2
* dgl-cuda >= 0.8.1
* torch-geometric >= 2.0.3