# +


import time
import torch
import argparse
import numpy as np



from statistics import mean, stdev
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.datasets import TUDataset


from bfi import BruteForceGraphIntegrator
from ftfi import TreeBasedGraphIntegrator
from ftfi import MinimumSpanningTreeConstructor


parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
                    default="MUTAG",
                    type=str,
                    help="Dataset name.")
parser.add_argument('--baseline', 
                    action='store_true')
parser.set_defaults(baseline=False)

args = parser.parse_args()

# -

def create_edge_lists(data, edge_weights=False, node_sim=False):
  # data: torch.pyg - edge attribute, data.num
  edge_list = []
  if edge_weights :
    all_weights = torch.argmax(data.edge_attr, dim=1) + 1
  
  for i in range(data.num_nodes):
    sublist = []
    # collect indices
    indices = (data.edge_index == i).nonzero(as_tuple=False)
    edge_ends_idx = indices[:,1][torch.where(indices[:,0]==0)]
    edge_ends = data.edge_index[1][edge_ends_idx]

    if (edge_weights is False) and (node_sim is False):
      for j in edge_ends :
        sublist.append([j.item(), 1.0])
    elif (edge_weights is True) and (node_sim is False) :
      correct_edge_wts = all_weights[indices[:,1][torch.where(indices[:,0]==0)]]
      for n, j in enumerate(edge_ends):
          sublist.append([j.item(), correct_edge_wts[n].item()])
    elif (edge_weights is False) and (node_sim is True):
      corr_wts = torch.nn.functional.cosine_similarity(data.x[i].unsqueeze(0), data.x[edge_ends]) + 1 #hack to not 0 cosine similarity
      for n, j in enumerate(edge_ends):
          sublist.append([j.item(), corr_wts[n].item()])
    else :
      raise NotImplementedError('Both edge attributes and node attributescan not be used')

    edge_list.append(sublist)
  return edge_list


def create_graph_feats(data,k_vals,func, choose_from, edge_weights=False, node_sim=False, symmetrize=False, threshold=6, baseline=False):
  # increase threshold for larger graphs

    left_eigs = []
    right_eigs = []

    graph = create_edge_lists(data, edge_weights=edge_weights, node_sim=node_sim)
    if baseline :
      tbgi = BruteForceGraphIntegrator(f_func=func, is_lambda=True, graph_adj_lists=graph)
      kernel_mat = tbgi.get_m_matrix()
    else :
      tbgi = TreeBasedGraphIntegrator(f_func=func, is_lambda=True, graph_adj_lists=graph,
                                    tree_constructor=MinimumSpanningTreeConstructor(),
                                    threshold=threshold) #hardcoded
      kernel_mat = tbgi.integrate(np.eye(data.num_nodes))
      if symmetrize :
        kernel_mat = (kernel_mat + kernel_mat.T)/2

    eigvals = np.real(np.linalg.eigvals(tbgi.integrate(np.eye(data.num_nodes))))
    if choose_from =='left' :
      left_eigs.append(np.sort(eigvals)[: k_vals]) #k_vals means a subset of the number of eigenvalues of the kernel matrix
    elif choose_from == 'right' :
      right_eigs.append(np.sort(eigvals)[-k_vals :])
    else :
      a = np.sort(eigvals)
      left_eigs.append(a[: k_vals]), right_eigs.append(a[-k_vals :])

    return left_eigs, right_eigs




dataset = TUDataset(root='../data/TUDataset', 
                    name=args.dataset)

# +
# Graph Classification



labels = [np.array(data.y) for data in dataset]

# best hyperparameters
rf_depths = [25]
eigs = [40]
threshold = 40
seeds = [1, 2, 3, 4, 5]

for num_eig in eigs :
  print(f'Number of eigen values: {num_eig}')  

  all_feats = []
  bad_idx = []

  start = time.time()
  for i, g in (enumerate(dataset)):
    try :
      all_feats.append(create_graph_feats(g, num_eig, 
                                          lambda x: x, 'left', 
                                          edge_weights=False, 
                                          node_sim=False, 
                                          symmetrize=True, 
                                          threshold=threshold, 
                                          baseline=args.baseline)[0])
    except:
      bad_idx.append(i)
  print(f'Feature construction time: {time.time() - start}')

  # needed to deal with small graphs to make sure we have the same length arrays
  for i,x in enumerate(all_feats):
    if len(x[0]) != num_eig:
      all_feats[i][0] = np.pad(x[0], (0, num_eig-len(x[0])))

  X = np.array(all_feats)
  all_labels = np.concatenate(labels).ravel()
  y = np.delete(labels, bad_idx)

  for seed in seeds:
      skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
      all_acc = []

      for d in rf_depths:
        print(f"depth is {d}")
        accu_stratified = []
        for train_index, test_index in skf.split(X, y):
          x_train_fold, x_test_fold = X[train_index], X[test_index]
          y_train_fold, y_test_fold = y[train_index], y[test_index]
    
          clf = RandomForestClassifier(max_depth=d, random_state=0)
          clf.fit(np.squeeze(x_train_fold), y_train_fold)
          accu_stratified.append(clf.score(np.squeeze(x_test_fold), y_test_fold))
    
        all_acc.append(accu_stratified)
        print('\nOverall Accuracy:', mean(accu_stratified)*100, '%')
        print('\nStandard Deviation is:', stdev(accu_stratified))
