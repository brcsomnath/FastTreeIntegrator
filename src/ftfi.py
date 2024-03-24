# +
import copy
import numpy as np

from collections import deque
from sklearn.model_selection import *

from gfi import GraphIntegrator



# -

class CompTree():
  """
  Low-level auxiliary functions for main auxiliary functions:

    1. integrate_on_tree,
    2. preprocess_tree,
    3. partition_tree,
    4. compute_struct_for_merge.

  """
  def __init__(self, left_child, right_child, left_id_sets, right_id_sets,
               left_distances, right_distances, left_ids, right_ids, bfgi):
    self.left_child = left_child
    self.right_child = right_child
    ### Fields containing content.
    self.left_id_sets = left_id_sets
    self.right_id_sets = right_id_sets
    self.left_distances = left_distances
    self.right_distances = right_distances
    self.left_ids = left_ids
    self.right_ids = right_ids
    self.bfgi = bfgi


# +
def find_vertices(tree, root):
  found_vertices = []
  parent = np.zeros(len(tree))
  parent[root] = -1
  queue = deque([root])
  while queue:
    m = queue.pop()
    for neighbour, weight in tree[m]:
      if neighbour != parent[m]:
        found_vertices.append(neighbour)
        queue.append(neighbour)
        parent[neighbour] = m
  return found_vertices


def bfs(graph, start):
  visited = np.zeros(len(graph))
  distances = np.zeros(len(graph))
  queue = deque([])
  def bfs_aux(graph, node, visited, distances, queue):
    visited[node] = 1
    queue.append(node)
    while queue:
      m = queue.pop()
      for neighbour, weight in graph[m]:
        if visited[neighbour] == 0:
          visited[neighbour] = 1
          distances[neighbour] = distances[m] + weight
          queue.append(neighbour)
    return distances
  return bfs_aux(graph, start, visited, distances, queue)


def dfs_subtree_sizes(tree, root):
  stack = []
  sizes = np.zeros(len(tree))
  discovered = np.zeros(len(tree))
  parent = np.zeros(len(tree), dtype=int)
  parent[root] = root
  stack.append(root)
  while len(stack) > 0:
    vertex = stack[-1]
    if not discovered[vertex]:
      sizes[vertex] += 1
      discovered[vertex] = 1
      count = 0
      for neighbor_weight in tree[vertex]:
        neighbor = neighbor_weight[0]
        if neighbor != parent[vertex]:
          count += 1
          stack.append(neighbor)
          parent[neighbor] = vertex
      if not count:
        sizes[parent[vertex]] += sizes[vertex]
        x = stack.pop()
    else:
      if vertex is not root:
        sizes[parent[vertex]] += sizes[vertex]
      x = stack.pop()
  return sizes


# +
class Level():
  def __init__(self, tf_shape):
    self.nodes = []
    if tf_shape is not None:
      self.tf_value = np.zeros(tf_shape)



def compute_cross_contribs(left_distances, 
                           left_tf_vals, 
                           right_distances,
                           right_tf_vals, 
                           f_func, 
                           is_lambda):
  if is_lambda:
    left_struct_matrix = np.zeros((len(left_distances), len(right_distances)))
    for i in range(len(left_distances)):
      for j in range(len(right_distances)):
        left_struct_matrix[i][j] = f_func(left_distances[i] + right_distances[j])
    right_struct_matrix = np.transpose(left_struct_matrix)
    cross_vals_for_left = np.einsum("kl,l...->k...", left_struct_matrix,
                                    np.array(right_tf_vals))
    cross_vals_for_right = np.einsum("lk,k...->l...", right_struct_matrix,
                                     np.array(left_tf_vals))
  else:
    a, b = f_func
    if (len(b) == 1 and b[0] == 1.0):
      l_res_shape = tuple([len(left_distances)] +
                          list(np.array(right_tf_vals).shape[1:]))
      r_res_shape = tuple([len(right_distances)] +
                          list(np.array(left_tf_vals).shape[1:]))
      cross_vals_for_left = np.zeros(l_res_shape)
      cross_vals_for_right = np.zeros(r_res_shape)
      for k in range(len(a)):
        for b in range(k + 1):
          l_array = np.array([np.power(l, b) for l in left_distances])
          r_array = np.array([np.power(r, k - b) for r in right_distances])
          renorm = a[k] * math.comb(k, b)
          cross_vals_for_left += renorm * np.einsum("n,m,m...->n...", l_array,
                                                    r_array,
                                                    np.array(right_tf_vals))
          cross_vals_for_right += renorm * np.einsum("m,n,n...->m...", r_array,
                                                     l_array,
                                                     np.array(left_tf_vals))

  return cross_vals_for_left, cross_vals_for_right


# -

def partition_tree(original_tree):  
  # Main auxiliary functions.

  root, pivot_point = 0, 0

  tree = copy.deepcopy(original_tree)

  parent = np.zeros(len(tree), dtype=int)
  parent[root] = -1
  sizes = dfs_subtree_sizes(tree, root)
  queue = deque([root])
  while queue:
    m = queue.pop()
    if sizes[m] > 0.5 * len(tree):
      pivot_point = m
    
    for neighbour, _ in tree[m]:
      if neighbour != parent[m]:
        queue.append(neighbour)
        parent[neighbour] = m

  sizes[parent[pivot_point]] = len(tree) - sizes[pivot_point]

  acc = 0
  index = -1
  for neighbor, _ in tree[pivot_point]:
    if acc > 0.25 * len(tree):
      break
    acc += sizes[neighbor]
    index += 1

  left_neighbors = copy.deepcopy(tree[pivot_point][:(index + 1)])
  right_neighbors = copy.deepcopy(tree[pivot_point][(index + 1):])
  tree[pivot_point] = left_neighbors
  left_vertex_set = find_vertices(tree, pivot_point)
  if_left_vertex = np.zeros(len(tree), dtype=int)

  for elem in left_vertex_set:
    if_left_vertex[elem] = 1

  left_tree = [left_neighbors]
  left_ids = [pivot_point]
  right_tree = [right_neighbors]
  right_ids = [pivot_point]
  for i in range(len(if_left_vertex)):
    if i == pivot_point:
      continue
    if if_left_vertex[i]:
      left_tree.append(copy.deepcopy(tree[i]))
      left_ids.append(i)
    else:
      right_tree.append(copy.deepcopy(tree[i]))
      right_ids.append(i)
  inv_left_ids = np.zeros(len(tree))
  inv_right_ids = np.zeros(len(tree))

  for i in range(len(left_ids)):
    inv_left_ids[left_ids[i]] = i

  for i in range(len(right_ids)):
    inv_right_ids[right_ids[i]] = i

  for i in range(len(left_tree)):
    for j in range(len(left_tree[i])):
      left_tree[i][j][0] = int(inv_left_ids[left_tree[i][j][0]])

  for i in range(len(right_tree)):
    for j in range(len(right_tree[i])):
      right_tree[i][j][0] = int(inv_right_ids[right_tree[i][j][0]])

  return [left_tree, left_ids, right_tree, right_ids]


def integrate_cross_terms(left_id_sets, 
                          right_id_sets, 
                          left_distances,
                          right_distances, 
                          f_func, 
                          is_lambda, 
                          X_tensor,
                          Y_tensor):
  left_tf_vals = []
  right_tf_vals = []
  for i in range(len(left_id_sets)):
    left_tf_vals.append(np.sum(X_tensor[left_id_sets[i],:], 
                               axis=0,
                               keepdims=False))

  for i in range(len(right_id_sets)):
    right_tf_vals.append(np.sum(X_tensor[right_id_sets[i],:], 
                                axis=0,
                                keepdims=False))

  res = compute_cross_contribs(left_distances, 
                               left_tf_vals, 
                               right_distances,
                               right_tf_vals, 
                               f_func, 
                               is_lambda)

  cross_vals_for_left = res[0]
  cross_vals_for_right = res[1]

  for i in range(len(cross_vals_for_left)):
    A = cross_vals_for_left[i]
    N = len(left_id_sets[i])
    fin_shape = tuple([N] + [1] * len(A.shape))
    Y_tensor[left_id_sets[i],:] += np.tile(A, fin_shape)
    
  for i in range(len(cross_vals_for_right)):
    A = cross_vals_for_right[i]
    N = len(right_id_sets[i])
    fin_shape = tuple([N] + [1] * len(A.shape))
    Y_tensor[right_id_sets[i],:] += np.tile(A, fin_shape)
  return Y_tensor


def compute_struct_for_merge(left_tree, left_ids, right_tree, right_ids):
  left_distances = bfs(left_tree, 0)
  right_distances = bfs(right_tree, 0)

  left_dict, right_dict = dict(), dict()

  for i in range(len(left_distances)):
    if left_distances[i] > 0.0:
      if left_distances[i] not in left_dict:
        left_dict[left_distances[i]] = Level(None)
      (left_dict[left_distances[i]].nodes).append(left_ids[i])

  for i in range(len(right_distances)):
    if right_distances[i] > 0.0:
      if right_distances[i] not in right_dict:
        right_dict[right_distances[i]] = Level(None)
      (right_dict[right_distances[i]].nodes).append(right_ids[i])

  left_dict_keys = list(left_dict.keys())
  left_dict_nodes = [x.nodes for x in list(left_dict.values())]
  right_dict_keys = list(right_dict.keys())
  right_dict_nodes = [x.nodes for x in list(right_dict.values())]
  return left_dict_keys, left_dict_nodes, right_dict_keys, right_dict_nodes


# +
def preprocess_tree(tree, f_func, is_lambda, threshold=6):
  if len(tree) < threshold:
    bfgi = BruteForceGraphIntegrator(f_func, is_lambda, tree)
    return CompTree(None, None, None, None, None, None, None, None, bfgi)
  else:
    left_tree, left_ids, right_tree, right_ids = partition_tree(tree)
    left_child = preprocess_tree(left_tree, 
                                 f_func, 
                                 is_lambda, 
                                 threshold)
    right_child = preprocess_tree(right_tree, 
                                  f_func, 
                                  is_lambda, 
                                  threshold)
    l_ds, l_ns, r_ds, r_ns = compute_struct_for_merge(
        left_tree, left_ids, right_tree, right_ids)
    return CompTree(left_child, 
                    right_child, 
                    l_ns, r_ns, 
                    l_ds, r_ds, 
                    left_ids, right_ids, 
                    None)


def integrate_on_tree(comp_tree, X_tensor, f_func, is_lambda):
  if comp_tree.bfgi is not None:
    return comp_tree.bfgi.integrate(X_tensor)
  else:
    left_result = integrate_on_tree(comp_tree.left_child,
                                    X_tensor[comp_tree.left_ids,:], f_func,
                                    is_lambda)
    right_result = integrate_on_tree(comp_tree.right_child,
                                     X_tensor[comp_tree.right_ids,:], f_func,
                                     is_lambda)
    Y_tensor = np.zeros_like(X_tensor)
    Y_tensor[comp_tree.left_ids,:] += left_result
    Y_tensor[comp_tree.right_ids,:] += right_result
    integrate_cross_terms(comp_tree.left_id_sets, comp_tree.right_id_sets,
                          comp_tree.left_distances, comp_tree.right_distances,
                          f_func, is_lambda, X_tensor, Y_tensor)
    return Y_tensor


# +

class TreeConstructor():  
  # Abstract class for the tree maker.
  def __init__(self):
    pass

  def construct_tree(graph_adj_lists):
    pass


class DisjointSet:
    # Minimum spanning tree functions.
    parent = {}
    size = {}
    
    def makeSet(self, n):
      for i in range(n):
        self.parent[i] = i
        self.size[i] = 1
        
    def find(self, k):
      if self.parent[k] == k:
        return k
      return self.find(self.parent[k])
    
    def union(self, a, b):
      x = self.find(a)
      y = self.find(b)
      if self.size[x] > self.size[y]:
        self.parent[y] = x
        self.size[x] += self.size[y]
      else:
        self.parent[x] = y
        self.size[y] += self.size[x]

    
def kruskal_algo(graph_adj_lists):
  mst = []
  tree = []
  N = len(graph_adj_lists)
  for i in range(N):
    tree.append([])
  ds = DisjointSet()
  ds.makeSet(N)
  index = 0
  edges = []
  for i in range(len(graph_adj_lists)):
    for j in range(len(graph_adj_lists[i])):
      if graph_adj_lists[i][j][0] < i:
        edges.append([i, graph_adj_lists[i][j][0], graph_adj_lists[i][j][1]])
  edges.sort(key=lambda x: x[2])
  while len(mst) != len(graph_adj_lists) - 1:
    src, dest, weight = edges[index]
    index = index + 1
    x = ds.find(src)
    y = ds.find(dest)
    if x != y:
      tree[src].append([dest, weight])
      tree[dest].append([src, weight])
      mst.append((src, dest, weight))
      ds.union(x, y)
  cost = sum([x[2] for x in mst])
  return tree


# +

class MinimumSpanningTreeConstructor(TreeConstructor):
  def __init__(self):
    pass
  def construct_tree(self, graph_adj_lists):
    return kruskal_algo(graph_adj_lists)


class TreeBasedGraphIntegrator(GraphIntegrator):
  def __init__(self, f_func, is_lambda, graph_adj_lists, tree_constructor,
               threshold=6):
    super().__init__(f_func, is_lambda, graph_adj_lists)
    self.tree = tree_constructor.construct_tree(graph_adj_lists)
    self.comp_tree = preprocess_tree(self.tree, f_func, is_lambda, threshold)
  def integrate(self, X_tensor, threshold=6):
    return integrate_on_tree(self.comp_tree, X_tensor, self.f_func,
                             self.is_lambda)
# -


