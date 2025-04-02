import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CADGroupingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        """
        Args:
            in_channels (int): Dimension of input node features.
            hidden_channels (int): Dimension for hidden layers.
            out_channels (int): Number of construction groups (each node can have multiple labels).
            num_layers (int): Total number of graph convolution layers.
            dropout (float): Dropout probability for the classifier.
        """
        super(CADGroupingGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        # First layer: input to hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Middle layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Last convolution layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Multi-label classifier head (using sigmoid activation)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)  # out_channels = number of construction groups
        )
        
    def forward(self, x, edge_index, batch):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # batch: [num_nodes] (batch assignment for each node)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # For per-node classification, we pass the node embeddings to the classifier.
        logits = self.classifier(x)
        # Use sigmoid for multi-label classification (each output is in [0,1])
        return torch.sigmoid(logits)
