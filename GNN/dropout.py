from typing import Optional, Tuple

import torch

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch import Tensor

from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor




# def dropout_node(edge_index: Tensor, p: float = 0.5,
#                  num_nodes: Optional[int] = None,
#                  training: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
#     """Randomly drops nodes from the adjacency matrix
#     :obj:`edge_index` with probability :obj:`p` using samples from
#     a Bernoulli distribution.

#     The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
#     indicating which edges were retained. (3) the node mask indicating
#     which nodes were retained.

#     Args:
#         edge_index (LongTensor): The edge indices.
#         p (float, optional): Dropout probability. (default: :obj:`0.5`)
#         num_nodes (int, optional): The number of nodes, *i.e.*
#             :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
#         training (bool, optional): If set to :obj:`False`, this operation is a
#             no-op. (default: :obj:`True`)

#     :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

#     Examples:

#         >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
#         ...                            [1, 0, 2, 1, 3, 2]])
#         >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
#         >>> edge_index
#         tensor([[0, 1],
#                 [1, 0]])
#         >>> edge_mask
#         tensor([ True,  True, False, False, False, False])
#         >>> node_mask
#         tensor([ True,  True, False, False])
#     """
#     if p < 0. or p > 1.:
#         raise ValueError(f'Dropout probability has to be between 0 and 1 '
#                          f'(got {p}')

#     num_nodes = maybe_num_nodes(edge_index, num_nodes)

#     if not training or p == 0.0:
#         node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
#         edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
#         return edge_index, edge_mask, node_mask

#     prob = torch.rand(num_nodes, device=edge_index.device)
#     node_mask = prob > p
#     edge_index, _, edge_mask = subgraph(node_mask, edge_index,
#                                         num_nodes=num_nodes,
#                                         return_edge_mask=True)
#     return edge_index, edge_mask, node_mask



def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    """Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask



# def dropout_path(edge_index: Tensor, p: float = 0.2, walks_per_node: int = 1,
#                  walk_length: int = 3, num_nodes: Optional[int] = None,
#                  is_sorted: bool = False,
#                  training: bool = True) -> Tuple[Tensor, Tensor]:
#     """Drops edges from the adjacency matrix :obj:`edge_index`
#     based on random walks. The source nodes to start random walks from are
#     sampled from :obj:`edge_index` with probability :obj:`p`, following
#     a Bernoulli distribution.

#     The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
#     indicating which edges were retained.

#     Args:
#         edge_index (LongTensor): The edge indices.
#         p (float, optional): Sample probability. (default: :obj:`0.2`)
#         walks_per_node (int, optional): The number of walks per node, same as
#             :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`1`)
#         walk_length (int, optional): The walk length, same as
#             :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`3`)
#         num_nodes (int, optional): The number of nodes, *i.e.*
#             :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
#         is_sorted (bool, optional): If set to :obj:`True`, will expect
#             :obj:`edge_index` to be already sorted row-wise.
#             (default: :obj:`False`)
#         training (bool, optional): If set to :obj:`False`, this operation is a
#             no-op. (default: :obj:`True`)

#     :rtype: (:class:`LongTensor`, :class:`BoolTensor`)

#     Example:

#         >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
#         ...                            [1, 0, 2, 1, 3, 2]])
#         >>> edge_index, edge_mask = dropout_path(edge_index)
#         >>> edge_index
#         tensor([[1, 2],
#                 [2, 3]])
#         >>> edge_mask # masks indicating which edges are retained
#         tensor([False, False,  True, False,  True, False])
#     """

#     if p < 0. or p > 1.:
#         raise ValueError(f'Sample probability has to be between 0 and 1 '
#                          f'(got {p}')

#     num_edges = edge_index.size(1)
#     edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
#     if not training or p == 0.0:
#         return edge_index, edge_mask

#     if random_walk is None:
#         raise ImportError('`dropout_path` requires `torch-cluster`.')

#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     edge_orders = None
#     ori_edge_index = edge_index
#     if not is_sorted:
#         edge_orders = torch.arange(num_edges, device=edge_index.device)
#         edge_index, edge_orders = sort_edge_index(edge_index, edge_orders,
#                                                   num_nodes=num_nodes)

#     row, col = edge_index
#     sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
#     start = row[sample_mask].repeat(walks_per_node)

#     deg = degree(row, num_nodes=num_nodes)
#     rowptr = row.new_zeros(num_nodes + 1)
#     torch.cumsum(deg, 0, out=rowptr[1:])
#     n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
#     e_id = e_id[e_id != -1].view(-1)  # filter illegal edges

#     if edge_orders is not None:
#         # permute edge ids
#         e_id = edge_orders[e_id]
#     edge_mask[e_id] = False
#     edge_index = ori_edge_index[:, edge_mask]

#     return edge_index, edge_mask