import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from dataset import CADGraphDataset

# Define the model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train():
    # Load dataset
    dataset = CADGraphDataset(root_dir="data/raw")
# or wherever your data is
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    if len(dataset) == 0:
        print("No data found in dataset. Check the data path and files.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the first sample to infer input feature size
    sample = dataset[0]
    in_channels = sample.x.size(1)
    hidden_channels = 64
    out_channels = 10  # Change as per your classification need

    # Initialize model
    model = GCN(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()  # Use BCEWithLogitsLoss or other if needed

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            # Dummy target for placeholder — adjust with real targets if available
            target = torch.zeros_like(out)

            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()

torch.save(model.state_dict(), "best_model.pt")


#print(f"Found {len(dataset)} files in dataset.")  # Should print a non-zero number
