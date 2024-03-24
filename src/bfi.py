# +
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import *

from gfi import *


# -

def compute_shortest_path_distances(graph_adj_lists):
  # Auxiliary functions for the brute-force integrator.
  N = len(graph_adj_lists)
  edges = np.zeros((N, N))
  for i in range(N):
    for j, w in graph_adj_lists[i]:
      edges[i,j] = w
      edges[j,i] = w
  csr_adjacency = csr_matrix(edges)
  return floyd_warshall(csgraph=csr_adjacency, directed=False)


def poly(x, coeff_list):
  accum = 0
  x_power = 1
  for i in range(len(coeff_list)):
    accum += x_power * coeff_list[i]
    x_power *= x
  return accum


class BruteForceGraphIntegrator(GraphIntegrator):
  def __init__(self, f_func, is_lambda, graph_adj_lists):
    super().__init__(f_func, is_lambda, graph_adj_lists)
    self.M = compute_shortest_path_distances(self.graph_adj_lists)
    for i in range(self.N):
      for j in range(self.N):
        if not self.is_lambda:
          numerator = poly(self.M[i][j], self.f_func[0])
          denominator = poly(self.M[i][j], self.f_func[1])
          self.M[i][j] = numerator / denominator
        else:
          self.M[i][j] = self.f_func(self.M[i][j])
  def integrate(self, X_tensor):
    return np.einsum("mn,n...->m...", self.M, X_tensor)
  def get_m_matrix(self):
    return self.M


