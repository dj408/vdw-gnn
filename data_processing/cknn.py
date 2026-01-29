"""
A continuous k-NN (CkNN) graph construction algorithm.

Citation: Berry, T. and Sauer, T. Consistent manifold representation for topological data analysis. Found. Data Sci. 1, 1-38 (2019).

arXiv: https://arxiv.org/pdf/1606.02353

"Theorem 2 states that the CkNN is the unique unweighted graph construction for which the (unnormalized) graph Laplacian converges spectrally to a Laplace-Beltrami operator on the manifold in the large data limit" (p. 2).

Code adapted from cknn package github (MIT license).
URL: https://github.com/chlorochrule/cknn/blob/master/cknn/cknn.py
Changes:
- Added use_raw_distances parameter to allow for raw distances to be used as weights.
- Export in PyG-friendly COO form (edge_index, edge_weight) instead of csr_matrix when is_sparse=True.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


def cknneighbors_graph(
    X,
    n_neighbors, 
    delta=1.0, 
    metric='euclidean', 
    t='inf',
    include_self=False, 
    is_sparse=True,
    return_instance=False,
    use_raw_distances=False,
) -> np.ndarray:

    cknn = CkNearestNeighbors(
        n_neighbors=n_neighbors, 
        delta=delta,
        metric=metric, 
        t=t, 
        include_self=include_self,
        is_sparse=is_sparse,
        use_raw_distances=use_raw_distances,
    )
    cknn.cknneighbors_graph(X)

    if return_instance:
        return cknn
    else:
        return cknn.ckng


class CkNearestNeighbors(object):
    """
    This object provides the all logic of CkNN.

    Args:
        n_neighbors: int, optional, default=5
            Number of neighbors to estimate the density around the point.
            It appeared as a parameter `k` in the paper.

        delta: float, optional, default=1.0
            A parameter to decide the radius for each points. The combination
            radius increases in proportion to this parameter.

        metric: str, optional, default='euclidean'
            The metric of each points. This parameter depends on the parameter
            `metric` of scipy.spatial.distance.pdist.

        t: 'inf' or float or int, optional, default='inf'
            The decay parameter of heat kernel. If t == 'inf', an unweighted adjacency is returned, unless use_raw_distances == True, then the raw distances are weights in a weighted adjacency matrix. If a float is given, the weights are calculated as follows:

                W_{ij} = exp(-(||x_{i}-x_{j}||^2) / t)

            For more infomation, see 'Laplacian Eigenmaps for
            Dimensionality Reduction and Data Representation' (Belkin, et. al).

        include_self: bool, optional, default=True
            All diagonal elements are 1.0 if this parameter is True.

        is_sparse: bool, optional, default=True
            The method `cknneighbors_graph` returns csr_matrix object if this
            parameter is True else returns ndarray object.

        use_raw_distances: bool, optional, default=False
            If True, the raw distances are used as weights.

        Returns:
            ckng: COO tuple (if self.is_sparse is True), or ndarray(if self.is_sparse is False)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        delta: float = 1.0,
        metric: str = 'euclidean',
        t: str = 'inf',
        include_self: bool = False,
        is_sparse: bool = True,
        use_raw_distances: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.metric = metric
        self.t = t
        self.include_self = include_self
        self.is_sparse = is_sparse
        self.ckng = None
        self.use_raw_distances = use_raw_distances


    def cknneighbors_graph(self, X):
        """
        A method to calculate the CkNN graph

        Args:
            X: ndarray
                The data matrix.

        return: tuple(edge_index, edge_weight) if self.is_sparse is True,
                where edge_index is shape (2, E) and edge_weight shape (E,).
                If self.is_sparse is False, returns a dense ndarray adjacency.
        """

        n_neighbors = self.n_neighbors
        delta = self.delta
        metric = self.metric
        t = self.t
        include_self = self.include_self
        is_sparse = self.is_sparse

        n_samples = X.shape[0]

        if n_neighbors < 1 or n_neighbors > n_samples-1:
            raise ValueError(
                "`n_neighbors` must be in the range 1 to number of samples"
            )
        if len(X.shape) != 2:
            raise ValueError("`X` must be 2D matrix")
        if n_samples < 2:
            raise ValueError("At least 2 data points are required")

        if metric == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("`X` must be square matrix")
            dmatrix = X
        else:
            dmatrix = squareform(pdist(X, metric=metric))

        darray_n_nbrs = np.partition(dmatrix, n_neighbors)[:, [n_neighbors]]
        ratio_matrix = dmatrix / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
        diag_ptr = np.arange(n_samples)

        if isinstance(delta, (int, float)):
            ValueError(
                "Invalid argument type: `delta` must be float or int",
            )
        adjacency = csr_matrix(ratio_matrix < delta)

        if include_self:
            adjacency[diag_ptr, diag_ptr] = True
        else:
            adjacency[diag_ptr, diag_ptr] = False

        if t == 'inf':
            if self.use_raw_distances:
                if is_sparse:
                    # Element-wise mask distances by adjacency
                    neigh = adjacency.multiply(dmatrix)
                else:
                    neigh = adjacency.toarray().astype(float) * dmatrix
            else:
                neigh = adjacency.astype(float)
        else:
            mask = adjacency.nonzero()
            weights = np.exp(-np.power(dmatrix[mask], 2) / t)
            dmatrix[:] = 0.
            dmatrix[mask] = weights
            neigh = csr_matrix(dmatrix)

        if is_sparse:
            # Export in PyG-friendly COO form (edge_index, edge_weight)
            coo = neigh.tocoo()
            edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)
            edge_weight = coo.data.astype(float)
            self.ckng = (edge_index, edge_weight)
        else:
            self.ckng = neigh.toarray()

        return self.ckng