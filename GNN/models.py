import torch
from torch_geometric.nn.conv import GCNConv, gcn_conv, PNAConv
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GNN.dropout import dropout_edge
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.utils import spmm
from GNN.conv_layers import GCNConv_reimplemented
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from typing import List, Callable
from torch import Tensor
import torch_scatter
import numpy as np
from torch.nn import Embedding
from torch.utils.data import DataLoader
from functools import partial
import networkx as nx
from torch_geometric.utils.sparse import index2ptr
from typing import Optional, Tuple
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data.data import Data
import torch.utils.checkpoint as checkpoint
from torch_geometric.utils import degree
from GNN.global_variables import global_vars as global_variables

MPNNConv = GCNConv if global_variables.model_to_use == "GCNConv" else PNAConv 


def heat_diffusion(edge_index: torch.Tensor, edge_weight: torch.Tensor, features_matrix: torch.Tensor, num_nodes: int, num_diffusion_steps: int=400) -> torch.Tensor:
    """
    Heat diffusion for imputed features from paper "On the Unreasonable Effectiveness of 
    Feature propagation in Learning on Graphs with Missing Node Features" by Rossi et al. 2021 
    <https://arxiv.org/abs/2111.12128> 
    """
    # Normalize the adjacency matrix without self-loops
    #edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    edge_index, edge_weight = gcn_conv.gcn_norm(edge_index, edge_weight, num_nodes, improved=False, add_self_loops=False, flow="source_to_target", dtype=None)
    # Create a sparse matrix from edge_index and edge_weight to use in heat diffusion
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        dtype=torch.float
    )  
    # Heat diffusion
    imputed_features = features_matrix.clone()
    row_original_features, col_original_features = torch.where(features_matrix != 0.0)
    
    for steps in range(0,num_diffusion_steps):
        out_old = imputed_features
        imputed_features = spmm(adj, imputed_features)
        imputed_features[row_original_features, col_original_features] = features_matrix[row_original_features, col_original_features]
        if steps%10==0:
            print(f"Average absolute difference from the preceeding step {(imputed_features - out_old).abs().sum().item()/num_nodes}")

    return imputed_features  


def page_rank(data, teleport_probs, damping_factor=0.85, max_iterations=100, tol=1e-6)-> torch.Tensor:
    """
    Modified PageRank algorithm with node-specific teleport probabilities.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data in the PyTorch Geometric format.
    teleport_probs : torch.Tensor
        Teleport probabilities for each node in the graph.
    damping_factor : float
        Probability of not teleporting to a random node.
    max_iterations : int
        Maximum number of iterations.
    tol : float
        Tolerance to determine algorithm convergence.

    Returns
    -------
    ranks : torch.FloatTensor
        PageRank scores for each node in the graph.
    """
    device = data.edge_index.device
    print("Device: ", device)
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = data.edge_weight

    # Normalize the weights, flow is from target to source
    edge_weight_row_sums = torch_scatter.scatter(edge_weight, edge_index[1], dim=0, dim_size=num_nodes, reduce='sum')
    edge_weight = edge_weight / edge_weight_row_sums[edge_index[1]]
    # Identify dangling nodes (nodes with no outgoing edges after applying edge weights)
    dangling_nodes = torch.where(edge_weight_row_sums == 0)[0]
    # Check if the edge weights are normalized
    edge_weight_row_sums = torch_scatter.scatter(edge_weight, edge_index[1], dim=0, dim_size=num_nodes, reduce='sum')
    edge_weight_row_sums[dangling_nodes] = 1
    assert torch.allclose(edge_weight_row_sums, torch.ones(num_nodes).to(device), atol=1e-3)

    # Create a sparse adjacency matrix
    adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    adj_matrix = adj_matrix.to_sparse_csr()

    ranks = (torch.ones(num_nodes) / num_nodes).to(device)

    # Power iteration
    for iteration in range(max_iterations):
        prev_ranks = ranks#.clone()
        #print("teleport_probs device: ", teleport_probs.device)
        #print("ranks device: ", ranks.device)
        #print("adj_matrix device: ", adj_matrix.device)
        #print("dangling_nodes device: ", dangling_nodes.device)
        #print("damping_factor device: ", torch.tensor(damping_factor).device)
        ranks = (1 - damping_factor) * teleport_probs + damping_factor * spmm(adj_matrix, ranks.unsqueeze(-1)).squeeze(-1) + torch.sum(prev_ranks[dangling_nodes]) * damping_factor / np.max([dangling_nodes.numel(), 1])
        if torch.allclose(prev_ranks, ranks, atol=tol):
            print("Converged! Iteration:", iteration)
            break

    return ranks
    
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(MLP, self).__init__()
        self.number_hidden_layers = len(hidden_channels)
        self.LinearLayers = torch.nn.ModuleList([])
        self.activation = torch.nn.LeakyReLU(0.1)
        if self.number_hidden_layers >= 1:
            self.LinearLayers.append(Linear(num_node_features, hidden_channels[0]))
        if self.number_hidden_layers >= 2:
            self.LinearLayers.append(Linear(hidden_channels[0], hidden_channels[1]))
        if self.number_hidden_layers >= 3:
            self.LinearLayers.append(Linear(hidden_channels[1], hidden_channels[2])) 
        if self.number_hidden_layers >= 4:
            self.LinearLayers.append(Linear(hidden_channels[2], hidden_channels[3]))     
    
        self.out = Linear(hidden_channels[-1], num_classes)
        
    def forward(self, x, drop_out_rate):
        for i in range(self.number_hidden_layers):
            x = self.LinearLayers[i](x) 
            x = self.activation(x) 
            x = F.dropout(x, p=drop_out_rate, training=self.training)   
        x = self.out(x)
        return x

    def reset_parameters(self):
        for layer in self.LinearLayers:          
            layer.reset_parameters()

class Node2Vec(torch.nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        edge_index (torch.Tensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        data: Data,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()

        self.random_walk_fn = torch.ops.pyg.random_walk

        if 'edge_weight' in data.keys:
            print("edge_weight")
            self.nx_graph = self.to_networkx(data.edge_index, data.edge_weight)
        else:
            print("no edge_weight")
            self.nx_graph = self.to_networkx(data.edge_index)

        self.edge_weighted_random_walk = BiasedRandomWalker(walk_length=5, p=1, q=1)

        self.num_nodes = maybe_num_nodes(data.edge_index, num_nodes)

        row, col = sort_edge_index(data.edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(self.num_nodes, embedding_dim,
                                   sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                          **kwargs)
    
    def to_networkx(self, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> nx.Graph:
        graph = nx.Graph()
        for index in range(edge_index.size(1)):
            source = edge_index[0][index].item()
            target = edge_index[1][index].item()
            if edge_weight is None:
                weight = 1.0
                graph.add_edge(source, target, weight=weight)
            else:
                weight = edge_weight[index].item()
                graph.add_edge(source, target, weight=weight)
        return graph

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw_edge_weights = torch.tensor(self.edge_weighted_random_walk.do_walks(self.nx_graph, batch))
        if not isinstance(rw_edge_weights, Tensor):
            rw_edge_weights = rw_edge_weights[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw_edge_weights[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        multi_class: str = 'auto',
        *args,
        **kwargs,
        ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')


"""
    random walk class for biased random walks taken from karateclub library:
        - https://github.com/benedekrozemberczki/karateclub/blob/master/karateclub/utils/walker.py
    The codebase has been adapted to work with pytorch geometric in the Node2Vec class.
    This random walk class takes into account the edge weights that are not considered in the
    original implementation of node2vec that uses torch.ops.torch_cluster.random_walk function.
"""
class BiasedRandomWalker:
    """
    Class to do biased second order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
        p (float): Return parameter (1/p transition probability) to move towards previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """

    walks: list
    graph: nx.classes.graph.Graph
    edge_fn: Callable
    weight_fn: Callable

    def __init__(self, walk_length: int, p: float, q: float):
        self.walk_length = walk_length
        self.p = p
        self.q = q

    def do_walk(self, node: int, graph: nx.Graph) -> List[str]:
        walk = [node]
        previous_node = None
        previous_node_neighbors = []
        for _ in range(self.walk_length - 1):
            current_node = walk[-1]
            edges = self.edge_fn(current_node)
            current_node_neighbors = np.array([edge[1] for edge in edges])

            weights = self.weight_fn(edges, graph)
            probability = np.piecewise(
                weights,
                [
                    current_node_neighbors == previous_node,
                    np.isin(current_node_neighbors, previous_node_neighbors),
                ],
                [lambda w: w / self.p, lambda w: w / 1, lambda w: w / self.q],
            )

            norm_probability = probability / sum(probability)
            selected = np.random.choice(current_node_neighbors, 1, p=norm_probability)[
                0
            ]
            walk.append(selected)

            previous_node_neighbors = current_node_neighbors
            previous_node = current_node

        return walk

    def do_walks(self, graph: nx.Graph, batch: Tensor) -> None:
        self.walks = []
        self.graph = graph
        self.edge_fn = _get_edge_fn(graph)
        self.weight_fn = _get_weight_fn(graph)

        for node in batch:
            walk_from_node = self.do_walk(node.item(), graph)
            self.walks.append(walk_from_node)
        return self.walks


def _undirected(node, graph) -> List[tuple]:
    edges = graph.edges(node)

    return edges


def _directed(node, graph) -> List[tuple]:
    edges = graph.out_edges(node, data=True)

    return edges


def _get_edge_fn(graph) -> Callable:
    fn = _directed if nx.classes.function.is_directed(graph) else _undirected

    fn = partial(fn, graph=graph)
    return fn


def _unweighted(edges: List[tuple]) -> np.ndarray:
    return np.ones(len(edges))


def _weighted(edges: List[tuple], graph: nx.Graph) -> np.ndarray:
    edges = list(edges)
    weights = map(lambda edge: graph.edges[edge[0], edge[1]]["weight"], edges)
    #weights = map(lambda edge: edge[-1]["weight"], edges)

    return np.array([*weights])


def _get_weight_fn(graph) -> Callable:
    fn = _weighted if nx.classes.function.is_weighted(graph) else _unweighted

    return fn

def get_filtered_features_matrix_and_encoder_layers_list(omics_to_use: str, features_matrix: torch.Tensor, latent_space_dimension: int = 128, fine_tuning_virus: str = "SARS-CoV-2"):
    #Define the list of omics to use
    omics_to_use_list = omics_to_use.split("-")
    #Define the list of linear layers
    linear_layers_list = torch.nn.ModuleList([])
    # Check the features matrix size
    print(f"Features matrix size {features_matrix.size()}")
    feature_matrix_expected_size = 0

    if "transcriptome" in omics_to_use_list:
        print("Transcriptome used")
        linear_layers_list.append(Linear(3, latent_space_dimension))
        feature_matrix_expected_size += 3

    if "proteome" in omics_to_use_list:
        print("Proteome used")
        linear_layers_list.append(Linear(3, latent_space_dimension))
        feature_matrix_expected_size += 3

    if fine_tuning_virus == "SARS-CoV-2":
        if "effectome" in omics_to_use_list:
            print("Effectome used")
            linear_layers_list.append(Linear(24, latent_space_dimension))
            feature_matrix_expected_size += 24

        if "interactome" in omics_to_use_list:
            print("Interactome used")
            linear_layers_list.append(Linear(24, latent_space_dimension))
            feature_matrix_expected_size += 24

    assert features_matrix.size()[1] >= feature_matrix_expected_size, "The features matrix size is not the expected one"

    return linear_layers_list, features_matrix.size()[1] - feature_matrix_expected_size

def standardize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / (std + 1e-6)

def random_permutation(features_matrix):
    num_samples, num_features = features_matrix.shape
    flattened_matrix = features_matrix.view(-1)
    permuted_indices = torch.randperm(flattened_matrix.size(0))
    permuted_matrix = flattened_matrix[permuted_indices].view(num_samples, num_features)
    return permuted_matrix

class MPNN(torch.nn.Module):
    def __init__(self, data, hidden_channels, jk_mode: str = "cat", input_pe_dimension_size=128,
                 input_gosemsim_dimension_size=128, input_prot_emb_dimension_size=480, fine_tuning_virus = "SARS-CoV-2", omics_to_use="transcriptome-proteome-effectome-interactome"):
        super(MPNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features_matrix = data.x.clone()
        # substitute the features matrix with a random permutation of the features matrix
        #self.features_matrix = random_permutation(self.features_matrix)
        #self.features_matrix = self.features_matrix[torch.randperm(self.features_matrix.size(0))]
        self.omics_features_matrix = data.x_ft.clone() 
        #self.omics_features_matrix = random_permutation(self.omics_features_matrix)

        self.is_fine_tuning = False
        self.is_gnn_explainer = False
        self.is_mlp_used = global_variables.is_mlp_used
        #self.is_message_weights_used = False
        self.num_conv_layers = len(hidden_channels)
        self.latent_space_dimension =  hidden_channels[0]*self.num_conv_layers if jk_mode == "cat" else hidden_channels[0] 

        # implement the jumpin knowledge layer
        if jk_mode != "no_jk":
            self.jumpin_knowledge_layer = JumpingKnowledge(mode=jk_mode, channels=hidden_channels[0], num_layers=self.num_conv_layers)
            self.jumpin_knowledge_layer_omics = JumpingKnowledge(mode=jk_mode, channels=hidden_channels[0], num_layers=self.num_conv_layers)
        else:
            self.jumpin_knowledge_layer = None
            self.jumpin_knowledge_layer_omics = None

        self.linear_from_latent_space = Linear(self.latent_space_dimension*2, self.latent_space_dimension)
        
        #Define the input omics matrix and the list of linear encoder layers
        self.linear_encoder_layers_list, input_pe_dimension_size_ft = get_filtered_features_matrix_and_encoder_layers_list(omics_to_use, self.omics_features_matrix, self.latent_space_dimension, fine_tuning_virus) 
        assert input_pe_dimension_size_ft == 0, "The input omics features matrix should not contain any positional encoding features"

        #Define the input features matrix and the list of linear encoder layers
        self.input_pe_dimension_size = input_pe_dimension_size
        print(f"Input positional encoding dimension size: {input_pe_dimension_size}")
        self.linear_pe_encoder_layer = Linear(input_pe_dimension_size, self.latent_space_dimension)
        self.input_pe_matrix = self.features_matrix[:, :input_pe_dimension_size].clone()
        self.input_gosemsim_dimension_size = input_gosemsim_dimension_size
        print(f"Input gosemsim encoding dimension size: {input_gosemsim_dimension_size}")
        self.linear_gosemsim_encoder_layer = Linear(input_gosemsim_dimension_size, self.latent_space_dimension)
        self.input_gosemsim_matrix = self.features_matrix[:, input_pe_dimension_size: input_pe_dimension_size + input_gosemsim_dimension_size].clone()
        print(f"Input protein embedding dimension size: {input_prot_emb_dimension_size}")
        self.linear_prot_emb_encoder_layer = Linear(input_prot_emb_dimension_size, self.latent_space_dimension) 
        self.input_prot_emb_matrix = self.features_matrix[:, input_pe_dimension_size + input_gosemsim_dimension_size: input_pe_dimension_size + input_gosemsim_dimension_size + input_prot_emb_dimension_size].clone()
        max_degree = -1
        
        if MPNNConv == PNAConv:
            print("PNAConv used") if not self.is_mlp_used else print("MLP used")

            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = d.max().item()

            # Compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long).to(self.device)
            deg += torch.bincount(d, minlength=deg.numel()).to(self.device)
        else:
            print("GCNConv used") if not self.is_mlp_used else print("MLP used")

        self.GCNConvs = torch.nn.ModuleList([])
        self.GCNConvs_omics = torch.nn.ModuleList([])
        if self.num_conv_layers >= 1:
            if MPNNConv == PNAConv:
                print("PNAConv used")
                self.GCNConvs.append(MPNNConv(self.latent_space_dimension, hidden_channels[0], aggregators=["mean", "min", "max", "std"], scalers=['identity', 'attenuation', 'amplification'], deg=deg))#'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'], deg=deg))
                self.GCNConvs_omics.append(MPNNConv(self.latent_space_dimension, hidden_channels[0], aggregators=["mean", "min", "max", "std"], scalers=['identity', 'attenuation', 'amplification'], deg=deg))#'identity', 'amplification', 'attenuation'], deg=deg))(
            else:
                self.GCNConvs.append(MPNNConv(self.latent_space_dimension, hidden_channels[0], improved=True)) #, cached=True))
                self.GCNConvs_omics.append(MPNNConv(self.latent_space_dimension, hidden_channels[0], improved=True)) #, cached=True))
        if self.num_conv_layers >= 2:
            if MPNNConv == PNAConv:
                self.GCNConvs.append(MPNNConv(hidden_channels[0], hidden_channels[1], aggregators=["mean", "min", "max", "std"], scalers=['identity', 'attenuation', 'amplification'], deg=deg))#, 'amplification', 'attenuation'], deg=deg)) 
                self.GCNConvs_omics.append(MPNNConv(hidden_channels[0], hidden_channels[1], aggregators=["mean", "min", "max", "std"], scalers=['identity', 'attenuation', 'amplification'], deg=deg))#, 'amplification', 'attenuation'], deg=deg))
            else:
                self.GCNConvs.append(MPNNConv(hidden_channels[0], hidden_channels[1], improved=True)) #, cached=True))
                self.GCNConvs_omics.append(MPNNConv(hidden_channels[0], hidden_channels[1], improved=True)) #, cached=True))
        if self.num_conv_layers >= 3:
            if MPNNConv == PNAConv:
                self.GCNConvs.append(MPNNConv(hidden_channels[1], hidden_channels[2], aggregators=["mean", "min", "max", "std"], scalers=['identity', 'attenuation', 'amplification'], deg=deg))
                self.GCNConvs_omics.append(MPNNConv(hidden_channels[1], hidden_channels[2], aggregators=["mean", "min", "max", "std"], scalers=['identity', 'attenuation', 'amplification'], deg=deg))
            else:
                self.GCNConvs.append(MPNNConv(hidden_channels[1], hidden_channels[2], improved=True)) #, cached=True))
                self.GCNConvs_omics.append(MPNNConv(hidden_channels[1], hidden_channels[2], improved=True)) #, cached=True))       
    
        
        self.linear_drug_targetable = Linear(hidden_channels[-1] * self.num_conv_layers, hidden_channels[-1])
        self.linear_disease_genes = Linear(hidden_channels[-1] * self.num_conv_layers, hidden_channels[-1])
            
        self.out_drug_targetable = Linear(hidden_channels[-1], 2)
        self.out_disease_genes = Linear(hidden_channels[-1], 2)             
        self.activation = torch.nn.LeakyReLU(0.1)
    #def init_message_weights(self, data: Data):
    #    # random initialize the message weights uniformly distibuted between -0.1 and 0.1 with 
    #    self.message_weights = Parameter(torch.rand(data.edge_weight.size(), device=self.device, dtype=torch.float32) * 0.2 - 0.1)

    def init_message_weights(self, data: Data):
        param = torch.rand(data.edge_weight.size(), device=self.device, dtype=torch.float32) * 0.4 - 0.2
        self.register_parameter('message_weights', Parameter(param))
    
    def reset_features_matrix_values(self, concat_features_matrix: torch.Tensor):
        concat_features_matrix = concat_features_matrix.clone()
        feature_matrix_dim = self.features_matrix.size()[1]
        omics_features_matrix_dim = self.omics_features_matrix.size()[1]

        # reset the class features matrices
        self.features_matrix = concat_features_matrix[:, :feature_matrix_dim]
        self.omics_features_matrix = concat_features_matrix[:, feature_matrix_dim: feature_matrix_dim + omics_features_matrix_dim]

        input_pe_dimension_size = self.input_pe_matrix.size()[1]
        self.input_pe_matrix = self.features_matrix[:, :input_pe_dimension_size]
        
        input_gosemsim_dimension_size = self.input_gosemsim_matrix.size()[1]       
        self.input_gosemsim_matrix = self.features_matrix[:, input_pe_dimension_size: input_pe_dimension_size + input_gosemsim_dimension_size]
        
        input_prot_emb_dimension_size = self.input_prot_emb_matrix.size()[1]
        self.input_prot_emb_matrix = self.features_matrix[:, input_pe_dimension_size + input_gosemsim_dimension_size: input_pe_dimension_size + input_gosemsim_dimension_size + input_prot_emb_dimension_size]
        

    def reset_parameters(self):
        self.linear_from_latent_space.reset_parameters()
        for layer in self.linear_encoder_layers_list:
            layer.reset_parameters()
        for conv in self.GCNConvs:
            conv.reset_parameters()
        for conv in self.GCNConvs_omics:
            conv.reset_parameters()
        self.out_drug_targetable.reset_parameters()
        self.out_disease_genes.reset_parameters()
        if self.jumpin_knowledge_layer is not None:
            self.jumpin_knowledge_layer.reset_parameters()
        if self.jumpin_knowledge_layer_omics is not None:
            self.jumpin_knowledge_layer_omics.reset_parameters()
        self.linear_prot_emb_encoder_layer.reset_parameters()
        self.linear_pe_encoder_layer.reset_parameters()
        self.linear_gosemsim_encoder_layer.reset_parameters()
        self.linear_drug_targetable.reset_parameters()
        self.linear_disease_genes.reset_parameters()

    def encode_omic_features(self):
        x = self.omics_features_matrix
        starting_input_features_position = 0
        x_latent_space_sum = torch.zeros((x.shape[0], self.latent_space_dimension), device=self.device)
        for _, layer in enumerate(self.linear_encoder_layers_list):
            x_latent_space_sum += layer(x[:, starting_input_features_position:starting_input_features_position + layer.in_features])
            starting_input_features_position += layer.in_features
            
        return x_latent_space_sum
    
    def forward(self, concat_features_matrix, edge_index, edge_weight, drop_out_rate, drop_out_rate_edges): 
        # extract the input features matrix that encode the positional encoding and pass them through a linear layer
        # the concatenated feature matrix is used only to reset the class features matrices needed when the model is used as an explainer
        if self.is_gnn_explainer:
            self.reset_features_matrix_values(concat_features_matrix)
            
        x_pe = self.input_pe_matrix
        x_pe = self.linear_pe_encoder_layer(x_pe)
        x_pe = self.activation(x_pe)

        x_go = self.input_gosemsim_matrix
        x_go = self.linear_gosemsim_encoder_layer(x_go)
        x_go = self.activation(x_go)

        x_prot_emb = self.input_prot_emb_matrix
        x_prot_emb = self.linear_prot_emb_encoder_layer(x_prot_emb)
        x_prot_emb = self.activation(x_prot_emb)


        x_jc: List[Tensor] = []
        # transform self.message_weights that in zeros when the message weight is < 0.0 and ones otherwise, then multiply the edge_weight by the message_weights

        #if self.is_message_weights_used and not self.is_gnn_explainer:
        #    edge_weight = torch.sigmoid(self.message_weights) #(edge_weight + torch.sigmoid(self.message_weights))/2 #  #torch.relu(torch.sign(self.message_weights))
        #    print(f"Message weights: {torch.sigmoid(self.message_weights[2000:2030])}")
        #indices = edge_weight > 0.0
        #edge_weight = edge_weight[indices]
        #edge_index = edge_index[:, indices]
        ## print some message weights

        if not self.is_mlp_used:
            for i in range(self.num_conv_layers):
                x_prot_emb = checkpoint.checkpoint(self.GCNConvs[i], x_prot_emb, edge_index, edge_weight, use_reentrant=False)
                x_prot_emb = self.activation(x_prot_emb) 
                # do not apply dropout_edge when the model is in evaluation mode
                if self.training:
                    edge_index, edge_mask = dropout_edge(edge_index, p=drop_out_rate_edges)
                    edge_weight = edge_weight[edge_mask]
                x_prot_emb = F.dropout(x_prot_emb, p=drop_out_rate, training=self.training)   
                x_jc.append(x_prot_emb)                                
        
            if self.jumpin_knowledge_layer is not None:
               x_prot_emb = self.jumpin_knowledge_layer(x_jc)

        if self.is_fine_tuning:
            x_omics = self.encode_omic_features()
            x_omics = self.activation(x_omics) if self.is_mlp_used else x_omics
            if not self.is_mlp_used:
                x_jc_omics: List[Tensor] = []
                for i in range(self.num_conv_layers):
                    x_omics = checkpoint.checkpoint(self.GCNConvs_omics[i], x_omics, edge_index, edge_weight, use_reentrant=False)
                    x_omics = self.activation(x_omics) 
                    # do not apply dropout_edge when the model is in evaluation mode
                    if self.training:
                        edge_index, edge_mask = dropout_edge(edge_index, p=drop_out_rate_edges)
                        edge_weight = edge_weight[edge_mask]
                    x_omics = F.dropout(x_omics, p=drop_out_rate, training=self.training)   
                    x_jc_omics.append(x_omics)

                if self.jumpin_knowledge_layer_omics is not None:
                    x_omics = self.jumpin_knowledge_layer_omics(x_jc_omics)
                
        # sum all the latent space vectors and zero out the ones that are not used
        if not global_variables.is_pe_used:
            print("PE not used")
            x_pe = torch.zeros_like(x_pe)
        if not global_variables.is_prot_emb_used:
            print("Prot emb not used")
            x_prot_emb = torch.zeros_like(x_prot_emb)
        if not global_variables.is_gosemsim_used:
            print("GO semsim not used")
            x_go = torch.zeros_like(x_go)

        x = x_pe + x_prot_emb

        if self.is_fine_tuning and global_variables.is_omics_used:
            x_go = x_go + x_omics 
        # concatenate x with x_go and x_pe
        x = torch.cat([x, x_go], dim=1)
        
        x = self.linear_from_latent_space(x)
        x = self.activation(x)

        ## Drug targetable prediction
        x_dt = self.linear_drug_targetable(x)
        x_dt = self.activation(x_dt)
        x_dt = self.out_drug_targetable(x_dt)
        # Disease genes prediction
        x_dg = self.linear_disease_genes(x)
        x_dg = self.activation(x_dg)
        x_dg = self.out_disease_genes(x_dg)
         
        return x_dg, x_dt
    
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta: 
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.path is not None:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    

class ModelExplainer(MPNN):
    def __init__(self, original_model, data, drop_out_rate, drop_out_rate_edges, device, is_drug_targets_output = False, fine_tuning_virus = "SARS-CoV-2"):
        super().__init__(data, original_model.hidden_channels, "cat", original_model.input_pe_dimension_size, original_model.input_gosemsim_dimension_size)
        # Copy the original model's parameters and buffers
        self.load_state_dict(original_model.state_dict())
        self.drop_out_rate = drop_out_rate
        self.drop_out_rate_edges = drop_out_rate_edges
        self.to(device)
        self.is_gnn_explainer = True
        self.is_fine_tuning = original_model.is_fine_tuning
        self.is_drug_targets_output = is_drug_targets_output

    def __call__(self, x, edge_index, edge_weight):
        # Call the original forward method with all required parameters
        # forward(self, x, edge_index, edge_weight, drop_out_rate, drop_out_rate_edges)
        out, out_dt = super().forward(x, edge_index, edge_weight, self.drop_out_rate, self.drop_out_rate_edges)
        # Apply softmax to the first element of the tuple to return the probabilities of the positive class
        if self.is_drug_targets_output:
            return F.softmax(out_dt, dim=1)[:,1]
        else:
            return F.softmax(out, dim=1)[:,1]
    
