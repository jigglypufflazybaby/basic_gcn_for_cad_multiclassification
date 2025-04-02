import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from model import CADGroupingGNN
from dataset import CADGraphDataset

def train():
    # Hyperparameters (adjust these as needed or load from config.py)
    in_channels = 128       # Example: dimension of your CAD face features
    hidden_channels = 256
    out_channels = 10       # Number of distinct construction groups (e.g., extrusions)
    num_layers = 3
    dropout = 0.5
    learning_rate = 0.001
    epochs = 100
    batch_size = 16

    # Load the dataset (assumes dataset folder structure: root/raw/ contains raw files)
    dataset = CADGraphDataset(root='data/cad_graph')
    # Split dataset into 80% training and 20% testing
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CADGroupingGNN(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Binary Cross Entropy loss for multi-label classification
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # Forward pass: obtain per-node multi-label predictions
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation on test data (here we simply compute an exact match accuracy per node)
    model.eval()
    total_correct = 0
    total_nodes = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # Threshold predictions at 0.5
            pred = (out > 0.5).float()
            # For each node, we check if all predicted labels match the ground truth exactly.
            total_correct += (pred == data.y).all(dim=1).sum().item()
            total_nodes += data.y.size(0)
    print("Test exact-match node accuracy:", total_correct / total_nodes)

if __name__ == '__main__':
    train()
