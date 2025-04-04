import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CADGroupingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge_types=3, num_layers=3, dropout=0.5):
        """
        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Hidden layer dimension.
            out_channels (int): Output labels (e.g. construction steps).
            num_edge_types (int): Number of unique edge relation types.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
        """
        super(CADGroupingGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.edge_embedding = torch.nn.Embedding(num_edge_types, 1)  # Maps relation type to edge weight

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Convert relation type (e.g., 0,1,2) into scalar edge weights
        edge_weight = self.edge_embedding(edge_attr).squeeze()  # Shape: [num_edges]
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)

        logits = self.classifier(x)
        return torch.sigmoid(logits)
