# Fast Tree-Field Integrators

This repository provides the implementation of the ICML submission titled "*Fast Tree-Field Integrators:
From Low Displacement Rank to Topological Transformers*".

## Installation

```
conda create -n ftfi python=3.8
source activate ftfi
pip install -r requirements.txt
```

## Graph Classification

To run FTFI for graph classification, use the following command:


```
cd src/
python graph_classifcation --dataset <dataset_name>
```

The dataset names used to run the above command is presented below:

```
['MUTAG', 
 'PTC_MR',
 'NC1',
 'COLLAB',
 'ENZYMES',
 'PROTEINS', 
 'IMDB-BINARY',
 'IMDB-MULTI',
 'REDDIT-BINARY', 
 'REDDIT-MULTI-5K', 
 'REDDIT-MULTI-12K']
```