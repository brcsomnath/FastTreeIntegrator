# +
from sklearn.model_selection import *


# -

class GraphIntegrator():
  def __init__(self, f_func, is_lambda, graph_adj_lists):
    """
    Args:
      * f_func: function of signature R -> R; either given as lambda or a a pair
          of two lists ([a_0, a_1,...,a_{t-1}], [b_0, b_1,..., b_{r-1}]). In the
          latter case, the lists encode a rational function:

             f(x) = (a_0 + a_1 * x + ... + a_{t-1} * x^{t-1}) /
                    (b_0 + b_1 * x + ... + b_{r-1} * x^{r-1})


      * is_lambda: boolean indicating whether f_func above is given as a lambda
          or a pair of two lists of coefficients (as described above)
      * graph_adj_lists: the adjacency lists encoding weighted undirected graph:
          graph_adj_lists[i][j] is a pair of the form (k, w), where k is the id
          of the jth neighbor of i (we start counting from 0) and w is the weight
          of an edge connecting i with k. We assume that graph nodes have
          identifiers: 0, 1, 2, ..., N-1, where N is the number of the nodes of
          the graph.
    """
    self.f_func = f_func
    self.is_lambda = is_lambda
    self.graph_adj_lists = graph_adj_lists
    self.N = len(graph_adj_lists)
  
  def integrate(self, X_tensor):
    """
        * X_tensor: tensor of the shape: N x b_1 x b_2 x ... b_s, where: N is the
          number of nodes of the graph and b_1, b_2, ... b_s are sizes of batch
          dimensions (arbitrary number of them).
        * Output: Tensor Y = einsum("mn,n...->m...", M, X_tensor), where M is the
          N x N matrix satisfying: M[i][j] ~ f_func(dist(i,j)) and dist is the
          shortest path distance between i and j in the graph.
    """
    pass


