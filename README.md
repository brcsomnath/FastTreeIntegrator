# Fast Tree-Field Integrators

This repository provides the implementation of the NeurIPS submission titled "*Fast Tree-Field Integrators:
From Low Displacement Rank to Topological Transformers*".

## Overview

We present a new class of fast polylog-linear algorithms based on the theory of structured matrices (in particular *low displacement rank*) for integrating tensor fields defined on weighted trees. Several applications of the resulting *fast tree-field integrators* (FTFIs) are presented, including: (a) approximation of graph metrics with tree metrics, (b) graph classification, (c) modeling on meshes, and finally (d) *Topological Transformers* (TTs) *TopVit* for images. For Topological Transformers, we propose  new relative position encoding (RPE) masking mechanisms with as few as **three** extra learnable parameters per Transformer layer, leading to **1.0-1.5\%+** accuracy gains. Importantly, most of FTFIs are **exact** methods, thus numerically equivalent to their brute-force counterparts. We show that when applied to graphs with thousands of nodes, those exact algorithms provide **5.7-13x** speedups. For completeness, we also propose approximate FTFI extensions, in particular via *Non-Uniform FFT* (NU-FFT) and random Fourier features (RFFs).

## Installation

```
conda create -n ftfi python=3.8
source activate ftfi
pip install -r requirements.txt
```

## Getting Started

We provide a standalone notebook to run our core FTFI algorithm in the `notebooks/` folder. This notebook demonstrates the functioning of FTFI and the baseline algorithm using demonstrative examples.  

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

## Vertex Normal Prediction

To run FTFI for vertex normal prediction on meshes, first set the ```meshgraph_path```, which is the input folder that stores mesh data, then use the following command:

```
cd src/
python vertex_normal_prediction.py
```



## Visualizations

We provide additional results and visualization of FTFI in several experimental settings in `visualizations/` folder. 
