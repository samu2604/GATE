
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv_reimplemented(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # If edge weights are available, use them in the normalization
        if edge_weight is not None:
            norm = norm * edge_weight

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

    class MPNN(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.lin = Linear(in_channels*2, out_channels, bias=False)
            self.bias = Parameter(torch.Tensor(out_channels))

            self.reset_parameters()

        def reset_parameters(self):
            self.lin.reset_parameters()
            self.bias.data.zero_()

        def forward(self, x, edge_index, edge_weight=None):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]
            # edge_weight has shape [E] or None

            # Step 1: Add self-loops to the adjacency matrix.
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

            # Step 2: Linearly transform node feature matrix.
            row, col = edge_index
            edge_features = torch.cat([x[row], x[col]], dim=1)
            edge_messages = self.lin(edge_features)

            # Step 3: Compute normalization.
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # If edge weights are available, use them in the normalization
            if edge_weight is not None:
                norm = norm * edge_weight

            # Step 4-5: Start propagating messages.
            out = self.propagate(edge_index, x=edge_messages, norm=norm)

            # Step 6: Apply a final bias vector.
            out += self.bias

            return out

        def message(self, x_j, norm):
            # x_j has shape [E, out_channels]

            # Step 4: Normalize node features.
            return norm.view(-1, 1) * x_j


