import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from dataset import CADGraphDataset
from model import CADGroupingGNN
import argparse
from config import config

def test(exp_dir):
    device = config['device']
    in_channels = config['in_channels']
    hidden_channels = config['hidden_channels']
    out_channels = config['out_channels']
    num_layers = config['num_layers']
    dropout = config['dropout']
    batch_size = config['batch_size']
    
    # Load dataset
    dataset = CADGraphDataset(root='data/cad_graph')
    test_dataset = dataset[int(0.8 * len(dataset)):]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CADGroupingGNN(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
    # Load the model checkpoint from exp_dir (modify if needed)
    checkpoint = torch.load(exp_dir + "/model_final.pth", map_location=device)
    model.load_state_dict(checkpoint)
    
    model.eval()
    total_correct = 0
    total_nodes = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # Threshold predictions at 0.5
            pred = (out > 0.5).float()
            total_correct += (pred == data.y).all(dim=1).sum().item()
            total_nodes += data.y.size(0)
    print("Test exact-match node accuracy:", total_correct / total_nodes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='runs/checkpoints', help='Directory of the experiment checkpoint')
    args = parser.parse_args()
    test(args.exp_dir)
